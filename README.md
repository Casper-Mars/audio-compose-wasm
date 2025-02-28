# WASM 多线程音频合成工具

这是一个使用Rust开发的WebAssembly音频合成工具，可以在浏览器中多线程下载和合成音频文件。

## 功能特点

- 多线程并行下载音频文件
- 按时间戳排序和合成音频
- 支持MP3和WAV等常见音频格式
- 通过WebAssembly在浏览器中高效运行
- 简单的JavaScript API接口

## 编译指南

### 前置要求

- Rust和Cargo (推荐使用rustup安装)
- wasm-pack (用于编译Rust到WebAssembly)

### 安装wasm-pack

```bash
cargo install wasm-pack
```

### 编译项目

```bash
# 在项目根目录下运行
wasm-pack build --target web
```

编译完成后，生成的WebAssembly文件将位于`pkg`目录中。

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

## 示例代码

详细的使用示例请参考项目中的`index.html`文件。

## 注意事项

- 由于浏览器的安全限制，音频URL必须支持CORS
- 大文件处理可能会占用较多内存，请根据实际情况调整使用方式
- 目前的实现是简单拼接音频数据，可能需要根据实际需求进行更复杂的音频处理