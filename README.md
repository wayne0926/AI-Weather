# 🌤️ AI Weather Assistant

一个智能的天气助手，结合了多个AI模型（Gemini和DeepSeek）的天气分析和建议功能，提供个性化的天气信息服务。

## ✨ 特色功能

- 🤖 双AI模型支持（Gemini/DeepSeek）提供智能天气分析
- 📍 多种定位方式（GPS/IP/城市名称）
- 🌈 实时天气信息和空气质量数据
- 📱 PWA支持，可安装到移动设备
- 🎨 自适应明暗主题
- 💡 智能天气建议
- 🔄 离线访问支持
- 📊 可调节的天气预测天数（1-14天）
- 💬 支持追问功能

## 🛠️ 技术栈

### 后端
- Python Flask
- OpenAI API (DeepSeek)
- Google Gemini API
- MeteoBule API (天气数据)
- WAQI API (空气质量数据)
- 百度地图API (地理位置服务)

### 前端
- HTML5
- CSS3 (现代化UI设计)
- JavaScript (原生)
- Service Worker (PWA支持)
- Marked.js (Markdown渲染)

## 🚀 快速开始

### 环境要求
- Python 3.11+
- 现代浏览器（支持PWA）

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
METEOBLUE_API_KEY=你的MeteoBule API密钥
WAQI_TOKEN=你的WAQI API令牌
DEEPSEEK_API_KEY=你的DeepSeek API密钥
BAIDU_MAP_AK=你的百度地图应用AK
BAIDU_MAP_SK=你的百度地图应用SK
GEMINI_API_KEY=你的Google Gemini API密钥
```

4. 运行应用：
```bash
python app.py
```

5. 访问应用：
打开浏览器访问 `http://localhost:5000`

## 📱 PWA安装

1. 使用支持PWA的浏览器访问应用
2. 等待浏览器提示"添加到主屏幕"
3. 点击安装即可将应用添加到设备

## 🎯 使用指南

1. **位置获取**
   - 点击📍按钮使用GPS定位
   - 输入城市名称手动定位
   - 自动IP定位（备选方案）

2. **AI模型选择**
   - Gemini：Google的新一代AI模型
   - DeepSeek：专业的中文AI模型

3. **天气查询**
   - 使用预设问题快速查询
   - 自定义问题获取详细建议
   - 调节预测天数（1-14天）
   - 支持追问功能深入了解

4. **主题切换**
   - 点击右上角🌙/☀️切换暗色/亮色主题
   - 支持系统主题自动切换

## 🔒 隐私说明

- 位置信息仅用于天气查询
- 不保存用户个人信息
- 支持离线访问
- API密钥安全存储

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