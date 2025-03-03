use anyhow::{anyhow, Result};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
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
    InternalError(String)
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
            self.buffer = download_audio(&self.url)
                .await
                .map_err(|e| AudioError::DownloadError(e.to_string()))?;
            Ok(())
        }
    }
}

#[wasm_bindgen]
pub struct AudioSynthesizer {
    segments: HashMap<String, audio::Segment>,
}

#[wasm_bindgen]
impl AudioSynthesizer {
    #[wasm_bindgen(constructor)]
    pub fn new(json_input: &str) -> Result<AudioSynthesizer, JsValue> {
        console_error_panic_hook::set_once();
        let segments: Vec<audio::Segment> = serde_json::from_str(json_input)
            .map_err(|e| AudioError::JsonParseError(e))?;

        let mut map = HashMap::new();
        for segment in segments {
            map.insert(segment.id.clone(), segment);
        }

        Ok(AudioSynthesizer { segments: map })
    }

    pub async fn add(&mut self, json_segment: &str) -> Result<(), JsValue> {
        let mut segment: audio::Segment = serde_json::from_str(json_segment)
            .map_err(|e| AudioError::JsonParseError(e))?;

        safe_progress_callback(0.0, "downloading");
        segment.download().await?;
        safe_progress_callback(100.0, "complete");

        self.segments.insert(segment.id.clone(), segment);
        Ok(())
    }

    pub fn delete(&mut self, id: &str) -> Result<(), JsValue> {
        self.segments.remove(id)
            .ok_or_else(|| AudioError::SegmentNotFound(id.to_string()))?;
        Ok(())
    }

    pub async fn update(&mut self, json_segment: &str) -> Result<(), JsValue> {
        let mut segment: audio::Segment = serde_json::from_str(json_segment)
            .map_err(|e| AudioError::JsonParseError(e))?;

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
    pub async fn compose(&self) -> Result<String, JsValue> {
        let mut segments: Vec<audio::Segment> = self.segments.values().cloned().collect();

        // 按开始时间排序
        segments.sort_by(|a, b| {
            let a_time = parse_timestamp(&a.start_time).unwrap_or(0);
            let b_time = parse_timestamp(&b.start_time).unwrap_or(0);
            a_time.cmp(&b_time)
        });

        // 并发合并音频数据
        const MERGE_BATCH_SIZE: usize = 10; // 每批处理10个片段
        let merge_tasks = segments.chunks(MERGE_BATCH_SIZE).map(|batch| {
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
        Ok(STANDARD.encode(&combined_audio))
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

// 初始化函数
#[wasm_bindgen(start)]
pub fn init() {
    console_log!("WASM Audio Synthesizer initialized");
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

// 下载音频文件
async fn download_audio(url: &str) -> Result<Vec<u8>> {
    let client = reqwest::Client::new();
    let response = client.get(url).send().await?;

    if !response.status().is_success() {
        return Err(anyhow!("Failed to download audio: {}", response.status()));
    }

    let bytes = response.bytes().await?;
    Ok(bytes.to_vec())
}
