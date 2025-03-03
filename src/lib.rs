use anyhow::Result;
use futures::future::join_all;
use serde::{Deserialize, Serialize};
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
    use std::io::Cursor;
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    #[derive(Serialize, Deserialize, Clone, Debug)]
    pub struct Segment {
        pub id: String,
        pub url: String,
        pub start_time: String,
        #[serde(skip)]
        pub buffer: Vec<u8>,
        #[serde(skip)]
        pub decoded_data: Option<(Vec<f32>, u32, u32)>,
    }

    impl Segment {
        // 解析开始时间为毫秒
        pub fn get_start_time_ms(&self) -> Result<u64, AudioError> {
            self.start_time
                .parse::<u64>()
                .map_err(|e| AudioError::TimestampError(format!("无法解析开始时间: {}", e)))
        }

        // 解码音频数据
        pub fn decode(&self, enable_logging: bool) -> Result<(Vec<f32>, u32, u32), AudioError> {
            if enable_logging {
                console_log!("开始解码音频数据，大小: {} 字节", self.buffer.len());
            }

            // 创建媒体源 - 克隆buffer以避免生命周期问题
            let buffer_clone = self.buffer.clone();
            let cursor = Cursor::new(buffer_clone);
            let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

            // 创建格式探测器
            let mut hint = Hint::new();
            if self.url.ends_with(".mp3") {
                hint.with_extension("mp3");
            } else if self.url.ends_with(".wav") {
                hint.with_extension("wav");
            }

            // 探测格式
            let format_opts = FormatOptions::default();
            let metadata_opts = MetadataOptions::default();
            let probed = symphonia::default::get_probe()
                .format(&hint, mss, &format_opts, &metadata_opts)
                .map_err(|e| {
                    AudioError::InternalError(format!("无法识别音频格式: {}, url: {}", e, self.url))
                })?;

            // 获取默认音轨
            let mut format = probed.format;
            let track = format
                .default_track()
                .ok_or_else(|| AudioError::InternalError("无法找到默认音轨".to_string()))?;

            // 创建解码器
            let mut decoder = symphonia::default::get_codecs()
                .make(&track.codec_params, &DecoderOptions::default())
                .map_err(|e| AudioError::InternalError(format!("无法创建解码器: {}", e)))?;

            // 获取采样率和声道数
            let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);
            let channels = track
                .codec_params
                .channels
                .unwrap_or(symphonia::core::audio::Channels::FRONT_LEFT)
                .count();

            if enable_logging {
                console_log!("音频信息: 采样率 {}Hz, {} 声道", sample_rate, channels);
            }

            // 解码所有帧
            let mut sample_data = Vec::new();
            let mut sample_buf = None;

            while let Ok(packet) = format.next_packet() {
                match decoder.decode(&packet) {
                    Ok(decoded) => {
                        // 获取或创建采样缓冲区
                        if sample_buf.is_none() {
                            sample_buf = Some(SampleBuffer::<f32>::new(
                                decoded.capacity() as u64,
                                *decoded.spec(),
                            ));
                        }

                        // 将解码后的音频数据转换为f32样本
                        if let Some(buf) = &mut sample_buf {
                            buf.copy_interleaved_ref(decoded);
                            sample_data.extend_from_slice(buf.samples());
                        }
                    }
                    Err(e) => {
                        if e.to_string().contains("EOF") {
                            break;
                        }
                        if enable_logging {
                            console_log!("解码帧时出错: {}", e);
                        }
                    }
                }
            }

            if enable_logging {
                console_log!("音频解码完成，共 {} 个样本", sample_data.len());
            }

            Ok((sample_data, sample_rate, channels as u32))
        }

        pub async fn download(&mut self, enable_logging: bool) -> Result<(), AudioError> {
            if enable_logging {
                console_log!("开始下载音频文件: {}", self.url);
            }
            let client = reqwest::Client::builder()
                .build()
                .map_err(|e| AudioError::DownloadError(e.to_string()))?;

            if enable_logging {
                console_log!("发送请求中...");
            }
            let response = client
                .get(&self.url)
                .send()
                .await
                .map_err(|e| AudioError::DownloadError(e.to_string()))?;
            if enable_logging {
                console_log!("响应了");
            }

            if !response.status().is_success() {
                return Err(AudioError::DownloadError(format!(
                    "Failed to download audio: {}",
                    response.status()
                )));
            }

            let bytes = response
                .bytes()
                .await
                .map_err(|e| AudioError::DownloadError(e.to_string()))?;
            self.buffer = bytes.to_vec();
            
            // 立即解码并缓存数据
            if enable_logging {
                console_log!("下载完成，开始解码音频数据");
            }
            
            match self.decode(enable_logging) {
                Ok(decoded) => {
                    // 存储解码后的数据
                    self.decoded_data = Some(decoded);
                    // 清空原始buffer以节省内存
                    if enable_logging {
                        console_log!("解码完成，释放原始数据缓冲区");
                    }
                    self.buffer.clear();
                    Ok(())
                }
                Err(e) => Err(e)
            }
        }
    }
}

#[wasm_bindgen]
pub struct AudioSynthesizer {
    segments: Vec<audio::Segment>,
    merge_batch_size: usize,
    download_batch_size: usize,
    enable_logging: bool,
}

#[wasm_bindgen]
impl AudioSynthesizer {
    #[wasm_bindgen(constructor)]
    pub fn new(
        json_input: &str,
        merge_batch_size: Option<usize>,
        download_batch_size: Option<usize>,
        enable_logging: Option<bool>,
    ) -> Result<AudioSynthesizer, JsValue> {
        console_error_panic_hook::set_once();
        let enable_logging = enable_logging.unwrap_or(false);
        if enable_logging {
            console_log!("AudioSynthesizer初始化开始，解析JSON数据...");
        }
        let segments: Vec<audio::Segment> =
            serde_json::from_str(json_input).map_err(|e| AudioError::JsonParseError(e))?;

        if enable_logging {
            console_log!("JSON解析完成，共有{}个音频片段", segments.len());
            console_log!("音频合成器创建完成，需要调用init方法下载音频");
        }

        Ok(AudioSynthesizer {
            segments,
            merge_batch_size: merge_batch_size.unwrap_or(20),
            download_batch_size: download_batch_size.unwrap_or(100),
            enable_logging,
        })
    }

    // 新增初始化方法，负责下载音频片段
    pub async fn init(&mut self) -> Result<(), JsValue> {
        let total_segments = self.segments.len();
        if self.enable_logging {
            console_log!("开始初始化，准备下载{}个音频片段...", total_segments);
        }

        if total_segments == 0 {
            if self.enable_logging {
                console_log!("没有音频片段需要下载");
            }
            return Ok(());
        }

        safe_progress_callback(0.0, "downloading");

        // 将所有下载任务分批处理
        let total_batches =
            (total_segments + self.download_batch_size - 1) / self.download_batch_size;

        for (batch_index, batch) in self
            .segments
            .chunks_mut(self.download_batch_size)
            .enumerate()
        {
            if self.enable_logging {
                console_log!(
                    "开始下载第{}/{}批音频片段...",
                    batch_index + 1,
                    total_batches
                );
            }

            // 创建当前批次的下载任务
            let download_tasks = batch
                .iter_mut()
                .map(|segment| segment.download(self.enable_logging));

            // 并行执行当前批次的下载任务
            let results = join_all(download_tasks).await;

            // 检查当前批次的下载结果
            for result in results {
                if let Err(e) = result {
                    if self.enable_logging {
                        console_log!("下载过程中发生错误: {}", e);
                    }
                    return Err(e.into());
                }
            }

            // 更新进度
            let progress = ((batch_index + 1) as f64 / total_batches as f64) * 100.0;
            safe_progress_callback(progress, "downloading");
        }

        safe_progress_callback(100.0, "complete");
        if self.enable_logging {
            console_log!("音频合成器初始化完成，可以开始合成音频");
        }

        Ok(())
    }

    pub async fn add(&mut self, json_segment: &str, pre_id: &str) -> Result<(), JsValue> {
        let mut segment: audio::Segment =
            serde_json::from_str(json_segment).map_err(|e| AudioError::JsonParseError(e))?;

        safe_progress_callback(0.0, "downloading");
        segment.download(self.enable_logging).await?;
        safe_progress_callback(100.0, "complete");

        // 如果pre_id为"-1"，则添加到数组首位
        if pre_id == "-1" {
            self.segments.insert(0, segment);
        } else {
            // 查找pre_id对应的位置
            let position = self
                .segments
                .iter()
                .position(|s| s.id == pre_id)
                .ok_or_else(|| AudioError::SegmentNotFound(pre_id.to_string()))?;
            // 在找到的位置后插入新片段
            self.segments.insert(position + 1, segment);
        }
        Ok(())
    }

    pub fn delete(&mut self, id: &str) -> Result<(), JsValue> {
        let position = self
            .segments
            .iter()
            .position(|segment| segment.id == id)
            .ok_or_else(|| AudioError::SegmentNotFound(id.to_string()))?;
        self.segments.remove(position);
        Ok(())
    }

    pub async fn update(&mut self, json_segment: &str) -> Result<(), JsValue> {
        let mut segment: audio::Segment =
            serde_json::from_str(json_segment).map_err(|e| AudioError::JsonParseError(e))?;

        let position = self
            .segments
            .iter()
            .position(|s| s.id == segment.id)
            .ok_or_else(|| AudioError::SegmentNotFound(segment.id.clone()))?;

        // 获取原有的音频片段
        let existing_segment = &self.segments[position];

        // 检查URL是否发生变化
        if existing_segment.url == segment.url {
            if self.enable_logging {
                console_log!("URL未变化，复用原有音频数据: {}", segment.url);
            }
            // 复用原有的音频数据
            segment.buffer = existing_segment.buffer.clone();
        } else {
            // URL已变化，需要重新下载
            if self.enable_logging {
                console_log!(
                    "URL已变化，重新下载音频: {} -> {}",
                    existing_segment.url,
                    segment.url
                );
            }
            safe_progress_callback(0.0, "downloading");
            segment.download(self.enable_logging).await?;
            safe_progress_callback(100.0, "complete");
        }

        // 更新数组中的元素
        self.segments[position] = segment;
        Ok(())
    }

    // 合成音频
    pub async fn compose(&self) -> Result<Box<[u8]>, JsValue> {
        if self.segments.is_empty() {
            if self.enable_logging {
                console_log!("没有音频片段需要合成");
            }
            return Ok(Box::new([]));
        }

        if self.enable_logging {
            console_log!("开始合成 {} 个音频片段", self.segments.len());
        }

        safe_progress_callback(0.0, "processing");

        // 第一步：使用已解码的音频数据
        let mut max_sample_rate = 24000;
        let mut max_channels = 1;
        let mut max_end_time_ms = 0;

        // 创建解码任务 - 对于未解码的片段进行解码，对于已解码的片段直接使用缓存数据
        let decode_tasks = self.segments.chunks(self.merge_batch_size).map(|batch| {
            let batch = batch.to_vec();
            let enable_logging = self.enable_logging;
            async move {
                let mut decoded_segments = Vec::new();
                for segment in batch {
                    // 检查是否有缓存的解码数据
                    if let Some((samples, sample_rate, channels)) = &segment.decoded_data {
                        if enable_logging {
                            console_log!("使用缓存的解码数据，无需重新解码");
                        }
                        let start_time_ms = segment.get_start_time_ms().unwrap_or(0);
                        decoded_segments.push((samples.clone(), *sample_rate, *channels, start_time_ms));
                    } else {
                        // 如果没有缓存数据，则需要解码
                        if enable_logging {
                            console_log!("未找到缓存数据，需要解码");
                        }
                        match segment.decode(enable_logging) {
                            Ok((samples, sample_rate, channels)) => {
                                let start_time_ms = segment.get_start_time_ms().unwrap_or(0);
                                decoded_segments.push((samples, sample_rate, channels, start_time_ms));
                            }
                            Err(e) => {
                                if enable_logging {
                                    console_log!("解码音频片段失败: {}, 跳过此片段", e);
                                }
                            }
                        }
                    }
                }
                decoded_segments
            }
        });

        // 并行执行所有解码任务
        let decode_results = join_all(decode_tasks).await;

        // 收集所有解码结果
        let mut all_decoded_segments = Vec::new();
        for result in decode_results {
            for (samples, sample_rate, channels, start_time_ms) in result {
                // 更新最大采样率和声道数
                if sample_rate > max_sample_rate {
                    max_sample_rate = sample_rate;
                }
                if channels > max_channels {
                    max_channels = channels;
                }

                // 计算结束时间
                let duration_ms =
                    (samples.len() as u64 * 1000) / (sample_rate as u64 * channels as u64);
                let end_time_ms = start_time_ms + duration_ms;
                if end_time_ms > max_end_time_ms {
                    max_end_time_ms = end_time_ms;
                }

                all_decoded_segments.push((samples, sample_rate, channels, start_time_ms));
            }
        }

        if self.enable_logging {
            console_log!(
                "所有音频片段解码完成，最大采样率: {}Hz, 最大声道数: {}, 总时长: {}ms",
                max_sample_rate,
                max_channels,
                max_end_time_ms
            );
        }

        // 计算输出缓冲区大小
        let total_samples = (max_end_time_ms as u32 * max_sample_rate * max_channels) / 1000;
        let mut output_buffer = vec![0.0f32; total_samples as usize];

        safe_progress_callback(30.0, "mixing");

        // 第二步：混合所有音频片段
        for (samples, sample_rate, channels, start_time_ms) in all_decoded_segments {
            // 计算起始样本位置
            let start_sample = (start_time_ms as u32 * max_sample_rate * max_channels) / 1000;

            // 如果需要重采样
            if sample_rate != max_sample_rate || channels != max_channels {
                // 简单重采样（实际项目中可能需要更复杂的重采样算法）
                let ratio = max_sample_rate as f32 / sample_rate as f32;
                let channel_ratio = max_channels as f32 / channels as f32;

                for i in 0..samples.len() / channels as usize {
                    let src_pos = i * channels as usize;
                    let dst_pos = (start_sample as usize)
                        + ((i as f32 * ratio) as usize * max_channels as usize);

                    if dst_pos + max_channels as usize <= output_buffer.len() {
                        for c in 0..channels as usize {
                            let dst_channel = (c as f32 * channel_ratio) as usize;
                            if dst_channel < max_channels as usize && src_pos + c < samples.len() {
                                output_buffer[dst_pos + dst_channel] += samples[src_pos + c];
                            }
                        }
                    }
                }
            } else {
                // 直接混合，无需重采样
                for i in 0..samples.len() {
                    let dst_pos = start_sample as usize + i;
                    if dst_pos < output_buffer.len() {
                        output_buffer[dst_pos] += samples[i];
                    }
                }
            }
        }

        safe_progress_callback(70.0, "encoding");

        // 第三步：归一化音频（防止削波）
        let mut max_amplitude = 0.0f32;
        for sample in &output_buffer {
            let abs_sample = sample.abs();
            if abs_sample > max_amplitude {
                max_amplitude = abs_sample;
            }
        }

        let scale_factor = if max_amplitude > 1.0 {
            1.0 / max_amplitude
        } else {
            1.0
        };
        for sample in &mut output_buffer {
            *sample *= scale_factor;
        }

        // 第四步：将浮点样本转换为16位PCM WAV格式
        let mut wav_data = Vec::new();

        // WAV头部
        let data_size = (output_buffer.len() * 2) as u32; // 16位 = 2字节/样本
        let file_size = 36 + data_size;

        // RIFF头
        wav_data.extend_from_slice(b"RIFF");
        wav_data.extend_from_slice(&(file_size as u32).to_le_bytes());
        wav_data.extend_from_slice(b"WAVE");

        // fmt子块
        wav_data.extend_from_slice(b"fmt ");
        wav_data.extend_from_slice(&(16u32).to_le_bytes()); // 子块大小
        wav_data.extend_from_slice(&(1u16).to_le_bytes()); // 音频格式 (PCM)
        wav_data.extend_from_slice(&(max_channels as u16).to_le_bytes()); // 声道数
        wav_data.extend_from_slice(&(max_sample_rate as u32).to_le_bytes()); // 采样率
        wav_data
            .extend_from_slice(&(max_sample_rate as u32 * max_channels as u32 * 2).to_le_bytes()); // 字节率
        wav_data.extend_from_slice(&(max_channels as u16 * 2).to_le_bytes()); // 块对齐
        wav_data.extend_from_slice(&(16u16).to_le_bytes()); // 位深度

        // data子块
        wav_data.extend_from_slice(b"data");
        wav_data.extend_from_slice(&(data_size as u32).to_le_bytes()); // 数据大小

        // 将浮点样本转换为16位PCM
        for sample in output_buffer {
            // 将浮点样本转换为16位整数
            let sample_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
            wav_data.extend_from_slice(&sample_i16.to_le_bytes());
        }

        safe_progress_callback(100.0, "complete");
        if self.enable_logging {
            console_log!("音频合成完成，总大小: {} 字节", wav_data.len());
        }

        Ok(wav_data.into_boxed_slice())
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
