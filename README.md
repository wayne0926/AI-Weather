# 🌤️ AI Weather Assistant

一个智能的天气助手，结合了多个AI模型（Gemini和DeepSeek）的天气分析和建议功能，提供个性化的天气信息服务。基于 FastAPI 的高性能异步架构，支持自然语言交互。

## ✨ 特色功能

- 🤖 双AI模型支持（Gemini和DeepSeek）提供智能天气分析
- 🧠 实时显示AI推理过程（DeepSeek模式）
- 📍 多种定位方式（GPS/IP/城市名称）
- 🌈 实时天气信息和空气质量数据
- 📱 PWA支持，可安装到移动设备
- 🎨 自适应明暗主题
- 💡 智能天气建议
- 📊 可调节的天气预测天数（1-14天）
- 💬 支持追问功能
- ⚡ 异步处理和性能优化
- 🔄 智能缓存机制

## 🛠️ 技术栈

### 后端
- FastAPI (异步 ASGI 框架)
- Uvicorn (高性能 ASGI 服务器)
- Pydantic (数据验证)
- aiohttp (异步 HTTP 客户端)
- OpenAI API (DeepSeek-R1-Distill-Qwen-32B)
- Google Gemini API (Gemini-2.0-Flash-Thinking)
- MeteoBule API (天气数据)
- WAQI API (空气质量数据)
- 百度地图API (地理位置服务)

### 前端
- Service Worker (PWA支持)
- Marked.js (Markdown渲染)

## 🚀 快速开始

### 环境要求
- Python 3.11+
- pip 包管理器

### 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/wayne0926/AI-Weather.git
cd AI-Weather
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置环境变量：
创建 `.env` 文件并设置以下API密钥：
```env
# MeteoBule API密钥 - 用于获取精确的天气预报数据
# 获取地址：https://content.meteoblue.com/en/access/packages
METEOBLUE_API_KEY=你的MeteoBule API密钥

# WAQI API令牌 - 用于获取空气质量数据
# 获取地址：https://aqicn.org/data-platform/token/
WAQI_TOKEN=你的WAQI API令牌

# Google Gemini API密钥 - 天气分析和建议（选项1）
# 获取地址：https://makersuite.google.com/app/apikey
GEMINI_API_KEY=你的Google Gemini API密钥

# DeepSeek API密钥 - 天气分析和建议（选项2）
# 获取地址：https://platform.deepseek.com/
DEEPSEEK_API_KEY=你的DeepSeek API密钥

# 百度地图应用AK和SK - 用于IP定位和地理编码服务
# 获取地址：https://lbsyun.baidu.com/apiconsole/key
BAIDU_MAP_AK=你的百度地图应用AK
BAIDU_MAP_SK=你的百度地图应用SK
```

> **注意事项：**
> - 所有API密钥都需要自行申请，免费额度通常足够个人使用
> - 建议在生产环境中使用环境变量或密钥管理服务
> - 某些API可能需要海外服务器才能访问

4. 运行应用：

开发环境：
```bash
uvicorn app:app --reload --port 8000
```

生产环境：
```bash
python app.py
```

5. 访问应用：
- 开发环境：打开浏览器访问 `http://localhost:8000`
- 生产环境：默认端口 8000，建议配置反向代理

## 🎯 使用指南

1. **位置获取**
   - 点击📍按钮使用GPS定位
   - 输入城市名称手动定位（由 Gemini-2.0-Flash-Lite 提供地理编码服务）

2. **AI模型选择**
   - Gemini：Google的新一代AI模型（Gemini-2.0-Flash-Thinking）
   - DeepSeek：专业的中文AI模型（DeepSeek-R1-Distill-Qwen-32B）
     - 选择DeepSeek模式时，可以看到AI的实时推理过程
     - 推理内容自动滚动显示，保持最新4行可见
     - 轻量化显示效果，不影响主要内容阅读

3. **天气查询**
   - 使用预设问题快速查询
   - 自定义问题获取详细建议
   - 调节预测天数（1-14天）
   - 支持追问功能深入了解

4. **主题切换**
   - 点击右上角🌙/☀️切换暗色/亮色主题
   - 支持系统主题自动切换

## 📡 API 接口

### 获取天气建议
```http
POST /get_weather_advice
Content-Type: application/json

{
    "latitude": 39.9042,
    "longitude": 116.4074,
    "query": "今天适合户外运动吗？",
    "forecast_days": 5,
    "model_type": "gemini"
}
```

### 追加提问
```http
POST /ask_followup
Content-Type: application/json

{
    "latitude": 39.9042,
    "longitude": 116.4074,
    "query": "那明天呢？",
    "forecast_days": 5,
    "model_type": "gemini"
}
```

### 获取IP位置
```http
GET /get_ip_location
```

### 获取城市位置
```http
POST /get_city_location
Content-Type: application/json

{
    "city": "北京"
}
```

## ⚙️ 生产环境优化

- 多工作进程（默认4个）
- uvloop 事件循环加速
- httptools 解析器
- 异步请求处理
- 智能缓存机制
- 请求参数验证
- 错误日志记录

## 🔒 隐私说明

- 位置信息仅用于天气查询
- 不保存用户个人信息
- API密钥安全存储
- 支持 CORS 安全配置

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 🙏 鸣谢

- [MeteoBule](https://www.meteoblue.com/)：天气数据支持
- [WAQI](https://waqi.info/)：空气质量数据支持
- [DeepSeek](https://deepseek.com/)：AI模型支持
- [Google Gemini](https://deepmind.google/technologies/gemini/)：AI模型支持
- [百度地图](https://lbsyun.baidu.com/)：地理位置服务支持
- [FastAPI](https://fastapi.tiangolo.com/)：高性能Web框架支持 