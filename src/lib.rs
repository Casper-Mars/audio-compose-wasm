use anyhow::{anyhow, Result};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use futures::future::join_all;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

// 定义音频片段结构
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AudioSegment {
    id: String,
    url: String,
    start_time: String, // 格式: "00:00:01,466"
    end_time: String,   // 格式: "00:00:02,828"
    #[serde(skip)]
    buffer: Vec<u8>, // 存储下载的音频数据
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

// 处理一批音频片段
async fn process_batch(batch: Vec<AudioSegment>) -> Result<Vec<u8>, anyhow::Error> {
    // 创建下载任务
    let download_tasks = batch.iter().map(|segment| {
        let url = segment.url.clone();
        async move {
            let buffer = download_audio(&url)
                .await
                .map_err(|e| anyhow!("Failed to download {}: {}", url, e))?;
            Ok::<(String, Vec<u8>), anyhow::Error>((url, buffer))
        }
    });

    // 并行下载所有音频
    let results: Vec<Result<(String, Vec<u8>), anyhow::Error>> = join_all(download_tasks).await;

    // 处理下载结果
    let mut download_map = std::collections::HashMap::new();
    for result in results {
        match result {
            Ok((url, buffer)) => {
                download_map.insert(url, buffer);
            }
            Err(e) => return Err(e),
        }
    }

    // 将下载的音频数据填充到对应的段中
    let mut segments = batch;
    for segment in &mut segments {
        if let Some(buffer) = download_map.get(&segment.url) {
            segment.buffer = buffer.clone();
        } else {
            return Err(anyhow!("Missing buffer for {}", segment.url));
        }
    }

    // 按开始时间排序
    segments.sort_by(|a, b| {
        let a_time = parse_timestamp(&a.start_time).unwrap_or(0);
        let b_time = parse_timestamp(&b.start_time).unwrap_or(0);
        a_time.cmp(&b_time)
    });

    // 合并音频数据
    let mut combined_audio = Vec::new();
    for segment in segments {
        combined_audio.extend(segment.buffer);
    }

    Ok(combined_audio)
}

// WASM导出函数：合成音频
#[wasm_bindgen]
pub async fn synthesize_audio(json_input: &str) -> Result<String, JsValue> {
    // 设置panic钩子，将Rust的panic转换为JavaScript错误
    console_error_panic_hook::set_once();

    // 解析输入JSON
    let segments: Vec<AudioSegment> = serde_json::from_str(json_input)
        .map_err(|e| JsValue::from(JsError::new(&format!("Failed to parse JSON: {}", e))))?;

    // 分批处理参数
    const BATCH_SIZE: usize = 50; // 每批处理50个音频片段
    let total_segments = segments.len();
    let mut combined_audio = Vec::new();

    // 分批处理音频片段
    for (batch_index, batch) in segments.chunks(BATCH_SIZE).enumerate() {
        let progress = (batch_index * BATCH_SIZE) as f64 / total_segments as f64 * 100.0;
        safe_progress_callback(progress, "downloading");

        let batch_result = process_batch(batch.to_vec())
            .await
            .map_err(|e| JsValue::from(JsError::new(&e.to_string())))?;

        combined_audio.extend(batch_result);
    }

    safe_progress_callback(100.0, "encoding");

    // 将合并后的音频数据转换为Base64字符串
    let base64_audio = STANDARD.encode(&combined_audio);

    safe_progress_callback(100.0, "complete");

    Ok(base64_audio)
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
