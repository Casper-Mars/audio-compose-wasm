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

// WASM导出函数：合成音频
#[wasm_bindgen]
pub async fn synthesize_audio(json_input: &str) -> Result<String, JsValue> {
    // 设置panic钩子，将Rust的panic转换为JavaScript错误
    console_error_panic_hook::set_once();

    // 解析输入JSON
    let mut segments: Vec<AudioSegment> = serde_json::from_str(json_input)
        .map_err(|e| JsValue::from(JsError::new(&format!("Failed to parse JSON: {}", e))))?;

    // 创建下载任务
    let download_tasks = segments.iter().map(|segment| {
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
            Err(e) => {
                return Err(JsValue::from(JsError::new(&format!(
                    "Download error: {}",
                    e
                ))));
            }
        }
    }

    // 将下载的音频数据填充到对应的段中
    for segment in &mut segments {
        if let Some(buffer) = download_map.get(&segment.url) {
            segment.buffer = buffer.clone();
        } else {
            return Err(JsValue::from(JsError::new(&format!(
                "Missing buffer for {}",
                segment.url
            ))));
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

    // 将合并后的音频数据转换为Base64字符串
    let base64_audio = STANDARD.encode(&combined_audio);

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
