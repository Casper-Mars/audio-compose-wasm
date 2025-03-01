# WASM 多线程音频合成工具

这是一个使用Rust开发的WebAssembly音频合成工具，可以在浏览器中多线程下载和合成音频文件。

## 功能特点

- 多线程并行下载音频文件
- 按时间戳排序和合成音频
- 支持MP3和WAV等常见音频格式
- 通过WebAssembly在浏览器中高效运行
- 简单的JavaScript API接口

## 安装指南

### 方式一：直接使用预编译包

1. 从 [Releases](https://github.com/Casper-Mars/audio-compose-wasm/releases) 页面下载最新的发布包
2. 解压下载的文件，将得到以下文件：
   - `wasm_bg.wasm`：WebAssembly 二进制文件
   - `wasm.js`：JavaScript 包装器
   - `wasm.d.ts`：TypeScript 类型定义（可选）

### 方式二：从源码编译

#### 前置要求

- Rust和Cargo (推荐使用rustup安装)
- wasm-pack (用于编译Rust到WebAssembly)

#### 安装wasm-pack

```bash
cargo install wasm-pack
```

#### 编译项目

```bash
# 在项目根目录下运行
wasm-pack build --target web
```

编译完成后，生成的WebAssembly文件将位于 pkg 目录中。

## 使用方法

### 在HTML中引入

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>WASM音频合成示例</title>
</head>
<body>
    <script type="module">
        import init, { synthesize_audio } from './pkg/wasm.js';
        
        async function run() {
            await init();
            
            // 音频片段数据
            const audioSegments = [
                {
                    "id": "1",
                    "url": "https://example.com/audio1.mp3",
                    "start_time": "00:00:01,466",
                    "end_time": "00:00:02,828"
                },
                {
                    "id": "2",
                    "url": "https://example.com/audio2.mp3",
                    "start_time": "00:00:02,308",
                    "end_time": "00:00:03,266"
                }
            ];
            
            try {
                // 调用WASM函数合成音频
                const base64Audio = await synthesize_audio(JSON.stringify(audioSegments));
                
                // 处理返回的Base64音频数据...
                console.log('音频合成完成');
            } catch (error) {
                console.error('音频合成失败:', error);
            }
        }
        
        run();
    </script>
</body>
</html>
```

### API说明

主要函数：`synthesize_audio(json_input: string): Promise<string>`

- 参数：包含音频片段信息的JSON字符串
- 返回：Base64编码的合成音频数据

输入JSON格式：
```json
[
  {
    "id": "唯一标识符",
    "url": "音频文件URL",
    "start_time": "开始时间 (格式: 00:00:01,466)",
    "end_time": "结束时间 (格式: 00:00:02,828)"
  },
  ...
]
```

#### 进度回调函数

可以通过在全局定义 `progress_callback` 函数来监听处理进度：

```javascript
window.progress_callback = function(percent, stage) {
    // percent: 处理进度（0-100）
    // stage: 当前阶段（'downloading' | 'encoding' | 'complete'）
    console.log(`Progress: ${percent}%, Stage: ${stage}`);
};
```

##### 回调函数参数说明：

- percent ：数值类型，表示当前进度（0-100）
- stage ：字符串类型，表示当前处理阶段
- 'downloading' ：正在下载音频文件
- 'encoding' ：正在编码合并后的音频
- 'complete' ：处理完成

##### 使用示例：

```javascript
window.progress_callback = function(percent, stage) {
    switch(stage) {
        case 'downloading':
            console.log(`下载进度：${percent}%`);
            break;
        case 'encoding':
            console.log('正在编码音频...');
            break;
        case 'complete':
            console.log('处理完成！');
            break;
    }
};
```

## 示例代码

详细的使用示例请参考项目中的`index.html`文件。

## 注意事项

- 由于浏览器的安全限制，音频URL必须支持CORS
- 大文件处理可能会占用较多内存，请根据实际情况调整使用方式
- 目前的实现是简单拼接音频数据，可能需要根据实际需求进行更复杂的音频处理