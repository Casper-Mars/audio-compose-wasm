use anyhow::{anyhow, Result};
use futures::future::join_all;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use wasm_bindgen::prelude::*;

#[derive(Error, Debug)]
pub enum AudioError {
    #[error("Failed to parse JSON: {0}")]
    JsonParseError(#[from] serde_json::Error),

    #[error("Failed to download audio: {0}")]
    DownloadError(String),

    #[error("Invalid timestamp format: {0}")]
    TimestampError(String),

    #[error("Segment not found: {0}")]
    SegmentNotFound(String),

    #[error("Internal error: {0}")]
    InternalError(String),
}

impl From<AudioError> for JsValue {
    fn from(error: AudioError) -> Self {
        JsValue::from(JsError::new(&error.to_string()))
    }
}

mod audio {
    use super::*;

    #[derive(Serialize, Deserialize, Clone, Debug)]
    pub struct Segment {
        pub id: String,
        pub url: String,
        pub start_time: String,
        pub end_time: String,
        #[serde(skip)]
        pub buffer: Vec<u8>,
    }

    impl Segment {
        pub async fn download(&mut self) -> Result<(), AudioError> {
            console_log!("开始下载音频文件: {}", self.url);
            let client = reqwest::Client::builder()
                .build()
                .map_err(|e| AudioError::DownloadError(e.to_string()))?;
                
            console_log!("发送请求中...");
            let response = client.get(&self.url)
                .send().await
                .map_err(|e| AudioError::DownloadError(e.to_string()))?;
            console_log!("响应了");
            
            if !response.status().is_success() {
                return Err(AudioError::DownloadError(format!("Failed to download audio: {}", response.status())));
            }

            let bytes = response.bytes().await
                .map_err(|e| AudioError::DownloadError(e.to_string()))?;
            self.buffer = bytes.to_vec();
            Ok(())
        }
    }
}

#[wasm_bindgen]
pub struct AudioSynthesizer {
    segments: HashMap<String, audio::Segment>,
    merge_batch_size: usize,
    download_batch_size: usize,
}

#[wasm_bindgen]
impl AudioSynthesizer {
    #[wasm_bindgen(constructor)]
    pub fn new(
        json_input: &str,
        merge_batch_size: Option<usize>,
        download_batch_size: Option<usize>,
    ) -> Result<AudioSynthesizer, JsValue> {
        console_error_panic_hook::set_once();
        console_log!("AudioSynthesizer初始化开始，解析JSON数据...");
        let segments: Vec<audio::Segment> =
            serde_json::from_str(json_input).map_err(|e| AudioError::JsonParseError(e))?;

        console_log!("JSON解析完成，共有{}个音频片段", segments.len());
        
        // 构建音频片段映射
        let mut map = HashMap::new();
        for segment in segments {
            map.insert(segment.id.clone(), segment);
        }
        console_log!("音频合成器创建完成，需要调用init方法下载音频");

        Ok(AudioSynthesizer {
            segments: map,
            merge_batch_size: merge_batch_size.unwrap_or(20),
            download_batch_size: download_batch_size.unwrap_or(100),
        })
    }
    
    // 新增初始化方法，负责下载音频片段
    pub async fn init(&mut self) -> Result<(), JsValue> {
        let total_segments = self.segments.len();
        console_log!("开始初始化，准备下载{}个音频片段...", total_segments);
        
        if total_segments == 0 {
            console_log!("没有音频片段需要下载");
            return Ok(());
        }
        
        safe_progress_callback(0.0, "downloading");

        // 将所有下载任务分批处理
        let mut segments_vec: Vec<_> = self.segments.values_mut().collect();
        let total_batches = (total_segments + self.download_batch_size - 1) / self.download_batch_size;
        
        for (batch_index, batch) in segments_vec.chunks_mut(self.download_batch_size).enumerate() {
            console_log!("开始下载第{}/{}批音频片段...", batch_index + 1, total_batches);
            
            // 创建当前批次的下载任务
            let download_tasks = batch.iter_mut().map(|segment| segment.download());
            
            // 并行执行当前批次的下载任务
            let results = join_all(download_tasks).await;
            
            // 检查当前批次的下载结果
            for result in results {
                if let Err(e) = result {
                    console_log!("下载过程中发生错误: {}", e);
                    return Err(e.into());
                }
            }
            
            // 更新进度
            let progress = ((batch_index + 1) as f64 / total_batches as f64) * 100.0;
            safe_progress_callback(progress, "downloading");
        }

        safe_progress_callback(100.0, "complete");
        console_log!("音频合成器初始化完成，可以开始合成音频");
        
        Ok(())
    }

    pub async fn add(&mut self, json_segment: &str) -> Result<(), JsValue> {
        let mut segment: audio::Segment =
            serde_json::from_str(json_segment).map_err(|e| AudioError::JsonParseError(e))?;

        safe_progress_callback(0.0, "downloading");
        segment.download().await?;
        safe_progress_callback(100.0, "complete");

        self.segments.insert(segment.id.clone(), segment);
        Ok(())
    }

    pub fn delete(&mut self, id: &str) -> Result<(), JsValue> {
        self.segments
            .remove(id)
            .ok_or_else(|| AudioError::SegmentNotFound(id.to_string()))?;
        Ok(())
    }

    pub async fn update(&mut self, json_segment: &str) -> Result<(), JsValue> {
        let mut segment: audio::Segment =
            serde_json::from_str(json_segment).map_err(|e| AudioError::JsonParseError(e))?;

        if !self.segments.contains_key(&segment.id) {
            return Err(AudioError::SegmentNotFound(segment.id.clone()).into());
        }

        safe_progress_callback(0.0, "downloading");
        segment.download().await?;
        safe_progress_callback(100.0, "complete");

        self.segments.insert(segment.id.clone(), segment);
        Ok(())
    }

    // 合成音频
    pub async fn compose(&self) -> Result<Box<[u8]>, JsValue> {
        let mut segments: Vec<audio::Segment> = self.segments.values().cloned().collect();

        // 按开始时间排序
        segments.sort_by(|a, b| {
            let a_time = parse_timestamp(&a.start_time).unwrap_or(0);
            let b_time = parse_timestamp(&b.start_time).unwrap_or(0);
            a_time.cmp(&b_time)
        });

        // 并发合并音频数据
        let merge_tasks = segments.chunks(self.merge_batch_size).map(|batch| {
            let batch = batch.to_vec();
            async move {
                let mut batch_audio = Vec::new();
                for segment in batch {
                    batch_audio.extend(segment.buffer);
                }
                batch_audio
            }
        });

        // 并行执行所有合并任务
        let merge_results = join_all(merge_tasks).await;

        // 按顺序组合所有批次的结果
        let mut combined_audio = Vec::new();
        for result in merge_results {
            combined_audio.extend(result);
        }

        safe_progress_callback(100.0, "complete");
        Ok(combined_audio.into_boxed_slice())
    }
}

// 定义进度回调函数类型
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = window)]
    fn progress_callback(percent: f64, stage: &str);
}

// 检查进度回调函数是否存在
fn has_progress_callback() -> bool {
    let window = web_sys::window().unwrap_or_else(|| panic!("No window object found"));
    js_sys::Reflect::has(&window, &JsValue::from_str("progress_callback")).unwrap_or(false)
}

// 安全地调用进度回调函数
fn safe_progress_callback(percent: f64, stage: &str) {
    if has_progress_callback() {
        progress_callback(percent, stage);
    }
}

// 添加控制台日志功能
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

// 辅助宏，用于在WASM中打印日志
#[macro_export]
macro_rules! console_log {
    ($($t:tt)*) => (log(&format!($($t)*)))
}

// 解析时间戳为毫秒
fn parse_timestamp(timestamp: &str) -> Result<u64> {
    let parts: Vec<&str> = timestamp.split(',').collect();
    if parts.len() != 2 {
        return Err(anyhow!("Invalid timestamp format"));
    }

    let time_parts: Vec<&str> = parts[0].split(':').collect();
    if time_parts.len() != 3 {
        return Err(anyhow!("Invalid time format"));
    }

    let hours: u64 = time_parts[0].parse()?;
    let minutes: u64 = time_parts[1].parse()?;
    let seconds: u64 = time_parts[2].parse()?;
    let milliseconds: u64 = parts[1].parse()?;

    Ok(hours * 3600000 + minutes * 60000 + seconds * 1000 + milliseconds)
}



// 初始化函数
#[wasm_bindgen(start)]
pub fn init() {
    console_log!("WASM Audio Synthesizer initialized");
}
