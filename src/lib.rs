use anyhow::Result;
use futures::future::join_all;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use wasm_bindgen::prelude::*;
use std::sync::{Arc, Mutex};

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
        pub start_time: u32,
        #[serde(skip)]
        pub buffer: Vec<u8>,
        #[serde(skip)]
        pub decoded_data: Option<(Vec<f32>, u32, u32)>,
    }

    impl Segment {
        // 解析开始时间为毫秒
        pub fn get_start_time_ms(&self) -> Result<u64, AudioError> {
            return Ok(self.start_time as u64);
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
                Err(e) => Err(e),
            }
        }
    }
}

/// 重采样音频数据的辅助函数
/// 
/// 将源音频数据按照目标采样率和通道数进行重采样，使用高质量的Sinc插值算法
pub fn resample_audio(
    samples: &[f32],
    src_sample_rate: u32,
    src_channels: u32,
    dst_sample_rate: u32,
    dst_channels: u32,
    enable_logging: bool,
) -> Vec<f32> {
    if enable_logging {
        console_log!("重采样音频: ({}Hz, {}通道) -> ({}Hz, {}通道)", 
            src_sample_rate, src_channels, dst_sample_rate, dst_channels);
    }
    
    // 如果源和目标参数相同，直接返回克隆的数据
    if src_sample_rate == dst_sample_rate && src_channels == dst_channels {
        return samples.to_vec();
    }
    
    // 首先处理通道转换，将多通道音频分离为单通道数据
    let mut channel_separated_data = Vec::with_capacity(src_channels as usize);
    for c in 0..src_channels as usize {
        let mut channel_data = Vec::with_capacity(samples.len() / src_channels as usize);
        for i in 0..(samples.len() / src_channels as usize) {
            let idx = i * src_channels as usize + c;
            if idx < samples.len() {
                channel_data.push(samples[idx]);
            }
        }
        channel_separated_data.push(channel_data);
    }
    
    // 对每个通道进行高质量重采样
    let mut resampled_channels = Vec::with_capacity(dst_channels as usize);
    let ratio = dst_sample_rate as f64 / src_sample_rate as f64;
    
    // 确定要使用的源通道
    let channels_to_use = std::cmp::min(src_channels, dst_channels) as usize;
    
    // 对每个需要的通道进行重采样
    for c in 0..channels_to_use {
        let src_channel = &channel_separated_data[c];
        let src_len = src_channel.len();
        let dst_len = (src_len as f64 * ratio) as usize;
        let mut dst_channel = Vec::with_capacity(dst_len);
        
        // 使用Sinc插值进行高质量重采样
        // Sinc窗口大小，较大的窗口提供更好的质量但计算成本更高
        let window_size = 16;
        
        for i in 0..dst_len {
            let src_idx_f = i as f64 / ratio;
            let src_idx = src_idx_f.floor() as isize;
            let frac = src_idx_f - src_idx as f64;
            
            let mut sum = 0.0;
            let mut weight_sum = 0.0;
            
            // 应用Sinc插值
            for j in -window_size..=window_size {
                let idx = src_idx + j;
                if idx >= 0 && idx < src_len as isize {
                    // Lanczos窗口函数
                    let x = std::f64::consts::PI * (frac - j as f64);
                    let mut weight = 1.0;
                    if x.abs() > 1e-6 { // 避免除以零
                        weight = (x.sin() / x) * ((x / window_size as f64).sin() / (x / window_size as f64));
                    }
                    
                    sum += src_channel[idx as usize] as f64 * weight;
                    weight_sum += weight;
                }
            }
            
            // 归一化结果
            if weight_sum > 1e-6 {
                dst_channel.push((sum / weight_sum) as f32);
            } else {
                dst_channel.push(0.0);
            }
        }
        
        resampled_channels.push(dst_channel);
    }
    
    // 如果目标通道数大于源通道数，复制现有通道
    while resampled_channels.len() < dst_channels as usize {
        let channel_to_copy = resampled_channels[resampled_channels.len() % channels_to_use].clone();
        resampled_channels.push(channel_to_copy);
    }
    
    // 将重采样后的通道数据交织在一起
    let mut resampled_data = Vec::with_capacity(resampled_channels[0].len() * dst_channels as usize);
    for i in 0..resampled_channels[0].len() {
        for c in 0..dst_channels as usize {
            if i < resampled_channels[c].len() {
                resampled_data.push(resampled_channels[c][i]);
            } else {
                resampled_data.push(0.0);
            }
        }
    }
    
    // 应用低通滤波器以消除混叠伪影
    if dst_sample_rate < src_sample_rate {
        resampled_data = apply_lowpass_filter(resampled_data, dst_channels, dst_sample_rate, src_sample_rate);
    }
    
    if enable_logging {
        console_log!("重采样完成，输出样本数: {}", resampled_data.len());
    }
    
    resampled_data
}

/// 应用低通滤波器以消除混叠伪影
fn apply_lowpass_filter(samples: Vec<f32>, channels: u32, sample_rate: u32, original_rate: u32) -> Vec<f32> {
    // 计算截止频率，通常为新采样率的一半（奈奎斯特频率）
    let cutoff_freq = 0.9 * (sample_rate as f32 / 2.0) / original_rate as f32;
    
    // 简单的FIR低通滤波器，使用汉宁窗
    let filter_size = 31; // 必须是奇数
    let half_size = filter_size / 2;
    
    // 创建滤波器系数
    let mut filter = vec![0.0; filter_size];
    let mut sum = 0.0;
    
    for i in 0..filter_size {
        let n = i as isize - half_size as isize;
        if n == 0 {
            filter[i] = 2.0 * cutoff_freq;
        } else {
            // Sinc函数
            filter[i] = (2.0 * cutoff_freq * std::f32::consts::PI * n as f32).sin() / (std::f32::consts::PI * n as f32);
        }
        
        // 应用汉宁窗
        let hanning = 0.5 + 0.5 * std::f32::consts::PI.cos() * (i as f32 / (filter_size - 1) as f32 * 2.0 - 1.0);
        filter[i] *= hanning;
        
        sum += filter[i];
    }
    
    // 归一化滤波器系数
    for i in 0..filter_size {
        filter[i] /= sum;
    }
    
    // 应用滤波器
    let mut filtered = vec![0.0; samples.len()];
    let channels = channels as usize;
    
    for c in 0..channels {
        for i in 0..(samples.len() / channels) {
            let mut sum = 0.0;
            for j in 0..filter_size {
                let n = i as isize - half_size as isize + j as isize;
                if n >= 0 && n < (samples.len() / channels) as isize {
                    sum += samples[n as usize * channels + c] * filter[j];
                }
            }
            filtered[i * channels + c] = sum;
        }
    }
    
    filtered
}

#[wasm_bindgen]
pub struct AudioSynthesizer {
    segments: Vec<audio::Segment>,
    merge_batch_size: usize,
    download_batch_size: usize,
    enable_logging: bool,
    total_time_ms: u32,
    memory_freed: bool,
    max_sample_rate: u32,
    max_channels: u32,
}

#[wasm_bindgen]
impl AudioSynthesizer {
    #[wasm_bindgen(constructor)]
    pub fn new(
        total_time_ms: u32,
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
            total_time_ms,
            memory_freed: false,
            max_sample_rate: 24000, // 默认采样率
            max_channels: 1,        // 默认通道数
        })
    }

    /// 更新最大采样率和通道数，并对不符合标准的音频片段进行重采样
    /// 
    /// 计算所有片段中的最大采样率和通道数，并根据需要对不符合标准的片段进行重采样
    /// 
    /// * `do_resample` - 是否执行重采样操作
    pub async fn update_max_audio_params_and_resample(&mut self, do_resample: bool) -> Result<(), JsValue> {
        if self.segments.is_empty() {
            return Ok(());
        }
        
        // 计算最大采样率和通道数
        if self.enable_logging {
            console_log!("开始计算最大采样率和通道数");
        }
        
        // 重置最大采样率和通道数
        self.max_sample_rate = 24000; // 默认值
        self.max_channels = 1;        // 默认值
        
        // 遍历所有片段，找出最大采样率和通道数
        for segment in &self.segments {
            if let Some((_, sample_rate, channels)) = &segment.decoded_data {
                self.max_sample_rate = self.max_sample_rate.max(*sample_rate);
                self.max_channels = self.max_channels.max(*channels);
            }
        }
        
        if self.enable_logging {
            console_log!("计算完成，最大采样率: {}Hz, 最大通道数: {}", self.max_sample_rate, self.max_channels);
        }
        
        // 如果不需要重采样，直接返回
        if !do_resample {
            return Ok(());
        }
        
        // 对不符合标准的音频片段进行重采样
        if self.enable_logging {
            console_log!("开始对不符合标准的音频片段进行重采样");
        }
        
        let mut resample_count = 0;
        let resample_batch_size = self.merge_batch_size; // 每批处理100个片段
        
        // 创建一个异步函数来处理单个片段的重采样
        async fn resample_segment(
            segment: &mut audio::Segment,
            max_sample_rate: u32,
            max_channels: u32,
            enable_logging: bool
        ) -> bool {
            if let Some((samples, sample_rate, channels)) = &segment.decoded_data {
                // 如果采样率或通道数不符合标准，需要重采样
                if *sample_rate != max_sample_rate || *channels != max_channels {
                    if enable_logging {
                        console_log!("重采样音频片段: {} ({}Hz, {}通道) -> ({}Hz, {}通道)", 
                            segment.id, sample_rate, channels, max_sample_rate, max_channels);
                    }
                    
                    // 使用通用重采样函数
                    let resampled_data = resample_audio(
                        samples, 
                        *sample_rate, 
                        *channels, 
                        max_sample_rate, 
                        max_channels,
                        enable_logging
                    );
                    
                    // 更新片段的解码数据
                    segment.decoded_data = Some((resampled_data, max_sample_rate, max_channels));
                    return true;
                }
            }
            false
        }
        
        // 将所有重采样任务分批处理
        let total_segments = self.segments.len();
        let total_batches = (total_segments + resample_batch_size - 1) / resample_batch_size;
        
        for (batch_index, batch) in self.segments.chunks_mut(resample_batch_size).enumerate() {
            if self.enable_logging {
                console_log!(
                    "开始重采样第{}/{}批音频片段...",
                    batch_index + 1,
                    total_batches
                );
            }
            
            // 创建当前批次的重采样任务
            let resample_tasks = batch
                .iter_mut()
                .map(|segment| resample_segment(
                    segment, 
                    self.max_sample_rate, 
                    self.max_channels, 
                    self.enable_logging
                ));
            
            // 并行执行当前批次的重采样任务
            let results = join_all(resample_tasks).await;
            
            // 统计重采样的数量
            resample_count += results.iter().filter(|&&result| result).count();
        }
        
        if self.enable_logging {
            if resample_count > 0 {
                console_log!("重采样完成，共处理了{}个音频片段", resample_count);
            } else {
                console_log!("所有音频片段均符合标准，无需重采样");
            }
        }
        
        Ok(())
    }

    // 新增初始化方法，负责下载音频片段
    pub async fn init(&mut self) -> Result<(), JsValue> {
        // 重置内存释放标志
        self.memory_freed = false;
        
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

        // 使用抽取出的方法计算最大采样率和通道数，并进行重采样
        self.update_max_audio_params_and_resample(true).await?;

        if self.enable_logging {
            console_log!("音频合成器初始化完成，可以开始合成音频");
        }

        safe_progress_callback(100.0, "complete");
        
        Ok(())
    }

    pub async fn add(&mut self, json_segment: &str, pre_id: &str) -> Result<(), JsValue> {
        // 检查内存是否已释放
        if self.memory_freed {
            return Err(AudioError::InternalError("内存已释放，请先调用init方法重新初始化".to_string()).into());
        }
        
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
        
        // 更新最大采样率和通道数，并进行重采样
        self.update_max_audio_params_and_resample(true).await?;
        
        Ok(())
    }

    /// 批量添加音频片段
    /// 
    /// 接收一个JSON数组字符串，结构与new方法的json_input参数一致，并将所有片段添加到指定位置
    /// 
    /// * `json_segments` - 包含多个音频片段的JSON数组字符串
    /// * `pre_id` - 在此ID对应的片段后插入新片段，如果为"-1"则添加到数组首位
    pub async fn add_batch(&mut self, json_segments: &str, pre_id: &str) -> Result<(), JsValue> {
        // 检查内存是否已释放
        if self.memory_freed {
            return Err(AudioError::InternalError("内存已释放，请先调用init方法重新初始化".to_string()).into());
        }
        
        // 解析JSON数组
        let mut segments: Vec<audio::Segment> =
            serde_json::from_str(json_segments).map_err(|e| AudioError::JsonParseError(e))?;

        if self.enable_logging {
            console_log!("开始批量添加{}个音频片段...", segments.len());
        }

        if segments.is_empty() {
            if self.enable_logging {
                console_log!("没有音频片段需要添加");
            }
            return Ok(());
        }

        // 开始下载进度
        safe_progress_callback(0.0, "downloading");

        // 并发下载所有片段
        let total_segments = segments.len();
        let total_batches = (total_segments + self.download_batch_size - 1) / self.download_batch_size;

        for (batch_index, batch) in segments.chunks_mut(self.download_batch_size).enumerate() {
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

        // 查找插入位置
        let insert_position = if pre_id == "-1" {
            0
        } else {
            // 查找pre_id对应的位置
            let position = self
                .segments
                .iter()
                .position(|s| s.id == pre_id)
                .ok_or_else(|| AudioError::SegmentNotFound(pre_id.to_string()))?;
            // 在找到的位置后插入新片段
            position + 1
        };

        // 批量插入所有片段
        for (i, segment) in segments.into_iter().enumerate() {
            self.segments.insert(insert_position + i, segment);
        }
        
        if self.enable_logging {
            console_log!("所有音频片段下载完成，开始更新音频参数并重采样");
        }

        // 更新最大采样率和通道数，并进行重采样
        self.update_max_audio_params_and_resample(true).await?;
        
        safe_progress_callback(100.0, "complete");
        
        if self.enable_logging {
            console_log!("批量添加音频片段完成");
        }
        
        Ok(())
    }

    pub async fn delete(&mut self, id: &str) -> Result<(), JsValue> {
        // 检查内存是否已释放
        if self.memory_freed {
            return Err(AudioError::InternalError("内存已释放，请先调用init方法重新初始化".to_string()).into());
        }
        
        let position = self
            .segments
            .iter()
            .position(|segment| segment.id == id)
            .ok_or_else(|| AudioError::SegmentNotFound(id.to_string()))?;
        self.segments.remove(position);
        
        // 更新最大采样率和通道数，并进行重采样
        self.update_max_audio_params_and_resample(true).await?;
        
        Ok(())
    }

    pub async fn update(&mut self, json_segment: &str) -> Result<(), JsValue> {
        // 检查内存是否已释放
        if self.memory_freed {
            return Err(AudioError::InternalError("内存已释放，请先调用init方法重新初始化".to_string()).into());
        }
        
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
            // 复用原有的音频数据和解码后的数据
            segment.buffer = existing_segment.buffer.clone();
            segment.decoded_data = existing_segment.decoded_data.clone();
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
        
        // 更新最大采样率和通道数，并进行重采样
        self.update_max_audio_params_and_resample(true).await?;
        
        Ok(())
    }

    /// 批量更新多个音频片段
    /// 
    /// # 参数
    /// * `json_segments` - JSON字符串，包含要更新的音频片段数组
    /// 
    /// # 返回值
    /// * `Ok(())` - 更新成功
    /// * `Err(JsValue)` - 更新失败，包含错误信息
    #[wasm_bindgen]
    pub async fn update_batch(&mut self, json_segments: &str) -> Result<(), JsValue> {
        // 检查内存是否已释放
        if self.memory_freed {
            return Err(AudioError::InternalError("内存已释放，请先调用init方法重新初始化".to_string()).into());
        }
        
        let segments: Vec<audio::Segment> =
            serde_json::from_str(json_segments).map_err(|e| AudioError::JsonParseError(e))?;

        if self.enable_logging {
            console_log!("开始批量更新{}个音频片段", segments.len());
        }

        let mut updated_count = 0;
        let total_count = segments.len();

        for mut segment in segments {
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
                // 复用原有的音频数据和解码后的数据
                segment.buffer = existing_segment.buffer.clone();
                segment.decoded_data = existing_segment.decoded_data.clone();
            } else {
                // URL已变化，需要重新下载
                if self.enable_logging {
                    console_log!(
                        "URL已变化，重新下载音频: {} -> {}",
                        existing_segment.url,
                        segment.url
                    );
                }
                
                // 计算下载进度
                let progress_start = (updated_count as f64 / total_count as f64) * 100.0;
                let progress_end = ((updated_count + 1) as f64 / total_count as f64) * 100.0;
                
                safe_progress_callback(progress_start, "downloading");
                segment.download(self.enable_logging).await?;
                safe_progress_callback(progress_end, "downloading");
            }

            // 更新数组中的元素
            self.segments[position] = segment;
            updated_count += 1;
            
            if self.enable_logging {
                console_log!("已更新 {}/{} 个音频片段", updated_count, total_count);
            }
        }
        
        // 更新最大采样率和通道数，并进行重采样
        self.update_max_audio_params_and_resample(true).await?;
        
        safe_progress_callback(100.0, "complete");
        
        if self.enable_logging {
            console_log!("批量更新完成，共更新{}个音频片段", updated_count);
        }
        
        Ok(())
    }

    /// 释放内存，清空所有音频片段的缓存数据
    /// 
    /// 在不再需要音频数据时调用此方法可以释放内存
    /// 调用此方法后，如果需要再次合成音频，需要重新调用init方法下载音频数据
    #[wasm_bindgen]
    pub fn free_memory(&mut self) -> Result<(), JsValue> {
        if self.enable_logging {
            console_log!("开始释放内存，清空{}个音频片段的缓存数据", self.segments.len());
        }

        // 遍历所有片段，清空buffer和decoded_data
        for segment in &mut self.segments {
            segment.buffer.clear();
            segment.decoded_data = None;
        }

        // 设置内存已释放标志
        self.memory_freed = true;

        if self.enable_logging {
            console_log!("内存释放完成");
        }
        
        Ok(())
    }

    // 合成音频
    pub async fn compose(&self) -> Result<Box<[u8]>, JsValue> {
        // 检查内存是否已释放
        if self.memory_freed {
            return Err(AudioError::InternalError("内存已释放，请先调用init方法重新初始化".to_string()).into());
        }
        
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

        use rayon::prelude::*;

        // 第一步：并行处理所有片段的基本信息
        if self.enable_logging {
            console_log!("开始并行处理所有音频片段的基本信息");
        }
        let (max_sample_rate, max_channels, max_end_time_ms, all_decoded_segments) = {
            let mut all_decoded_segments = Vec::new();
            let mut max_sample_rate = 24000;
            let mut max_channels = 1;
            let mut max_end_time_ms = 0;

            // 并行迭代处理每个片段
            let segments_info: Vec<_> = self
                .segments
                .par_iter()
                .filter_map(|segment| {
                    if let Some((samples, sample_rate, channels)) = &segment.decoded_data {
                        let start_time_ms = segment.get_start_time_ms().unwrap_or(0);
                        let duration_ms = (samples.len() as u64 * 1000)
                            / (*sample_rate as u64 * *channels as u64);
                        Some((
                            *sample_rate,
                            *channels,
                            start_time_ms,
                            duration_ms,
                            samples.clone(),
                        ))
                    } else {
                        None
                    }
                })
                .collect();

            // 处理收集到的信息
            for (sample_rate, channels, start_time_ms, duration_ms, samples) in segments_info {
                max_sample_rate = max_sample_rate.max(sample_rate);
                max_channels = max_channels.max(channels);
                let end_time_ms = start_time_ms + duration_ms;
                max_end_time_ms = max_end_time_ms.max(end_time_ms);
                all_decoded_segments.push((samples, sample_rate, channels, start_time_ms));
            }

            (
                max_sample_rate,
                max_channels,
                max_end_time_ms,
                all_decoded_segments,
            )
        };

        if self.enable_logging {
            console_log!(
                "所有音频片段解码完成，最大采样率: {}Hz, 最大声道数: {}, 总时长: {}ms",
                max_sample_rate,
                max_channels,
                max_end_time_ms
            );
        }

        // 使用传入的总时长计算输出缓冲区大小
        let total_samples = ((self.total_time_ms as u64 * max_sample_rate as u64 * max_channels as u64) / 1000) as usize;
        let output_buffer: Arc<Vec<Mutex<f32>>> = Arc::new((0..total_samples).map(|_| Mutex::new(0.0f32)).collect());

        safe_progress_callback(30.0, "mixing");

        // 第二步：并行处理所有音频片段，直接写入共享输出缓冲区
        if self.enable_logging {
            console_log!("开始并行混合所有音频片段");
        }

        // 将所有片段分批处理，每批处理merge_batch_size个片段
        for batch in all_decoded_segments.chunks(self.merge_batch_size) {
            // 对每批片段并行处理
            batch.par_iter().for_each(|(samples, sample_rate, channels, start_time_ms)| {
                // 计算起始样本位置
                let start_sample = (start_time_ms * max_sample_rate as u64 * max_channels as u64) / 1000;

                // 如果需要重采样
                if *sample_rate != max_sample_rate || *channels != max_channels {
                    // 使用通用重采样函数获取重采样后的数据
                    let resampled = resample_audio(
                        samples, 
                        *sample_rate, 
                        *channels, 
                        max_sample_rate, 
                        max_channels,
                        self.enable_logging
                    );
                    
                    // 将重采样后的数据添加到输出缓冲区
                    for i in 0..resampled.len() {
                        let dst_pos = start_sample as usize + i;
                        if dst_pos < total_samples {
                            let mut sample = output_buffer[dst_pos].lock().unwrap();
                            *sample += resampled[i];
                        }
                    }
                } else {
                    // 直接混合，无需重采样
                    for i in 0..samples.len() {
                        let dst_pos = start_sample as usize + i;
                        if dst_pos < total_samples {
                            let mut sample = output_buffer[dst_pos].lock().unwrap();
                            *sample += samples[i];
                        }
                    }
                }
            });

            // 在每批处理完成后可以添加一些进度更新或其他操作
            if self.enable_logging {
                console_log!("完成一批音频片段的混合处理");
            }
        }

        let mut output_buffer: Vec<f32> = Arc::try_unwrap(output_buffer)
            .unwrap_or_else(|_| panic!("Failed to unwrap Arc"))
            .iter()
            .map(|mutex| *mutex.lock().unwrap())
            .collect();

        safe_progress_callback(70.0, "encoding");

        // 第三步：并行计算最大振幅
        if self.enable_logging {
            console_log!("开始并行计算最大振幅");
        }
        let max_amplitude = output_buffer
            .par_iter()
            .map(|sample| sample.abs())
            .reduce(|| 0.0f32, f32::max);

        let scale_factor = if max_amplitude > 1.0 {
            1.0 / max_amplitude
        } else {
            1.0
        };

        // 并行应用缩放因子
        output_buffer.par_iter_mut().for_each(|sample| {
            *sample *= scale_factor;
        });

        // 第四步：将浮点样本转换为16位PCM WAV格式
        if self.enable_logging {
            console_log!("开始将浮点样本转换为16位PCM WAV格式");
        }
        let mut wav_data = Vec::new();

        // WAV头部
        let data_size = if output_buffer.len() * 2 > u32::MAX as usize {
            return Err(
                AudioError::InternalError("音频数据太大，超过WAV格式限制".to_string()).into(),
            );
        } else {
            (output_buffer.len() * 2) as u32
        };
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
                                                                             // 计算字节率时避免溢出
        let byte_rate = if (max_sample_rate as u64 * max_channels as u64 * 2) > u32::MAX as u64 {
            return Err(AudioError::InternalError(
                "采样率或声道数过大，超过WAV格式限制".to_string(),
            )
            .into());
        } else {
            (max_sample_rate * max_channels * 2) as u32
        };
        wav_data.extend_from_slice(&byte_rate.to_le_bytes());
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
