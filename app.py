# backend.py
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, AsyncGenerator
import aiohttp
import json
from openai import OpenAI
import os
import hashlib
from dotenv import load_dotenv
import time
from google import genai
import logging
from functools import wraps
from pathlib import Path
import uvicorn
import asyncio

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 初始化天气缓存字典
weather_cache: Dict[str, Any] = {}

# 加载环境变量
load_dotenv()

# 创建FastAPI应用
app = FastAPI(
    title="Weather AI Assistant",
    description="An AI-powered weather assistant API",
    version="1.0.0",
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 环境变量
METEOBLUE_API_KEY = os.getenv("METEOBLUE_API_KEY")
WAQI_TOKEN = os.getenv("WAQI_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
BAIDU_MAP_AK = os.getenv("BAIDU_MAP_AK")
BAIDU_MAP_SK = os.getenv("BAIDU_MAP_SK")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# API密钥验证
required_keys = {
    'METEOBLUE_API_KEY': METEOBLUE_API_KEY,
    'WAQI_TOKEN': WAQI_TOKEN,
    'DEEPSEEK_API_KEY': DEEPSEEK_API_KEY,
    'BAIDU_MAP_AK': BAIDU_MAP_AK,
    'BAIDU_MAP_SK': BAIDU_MAP_SK,
    'GEMINI_API_KEY': GEMINI_API_KEY
}

missing_keys = [key for key, value in required_keys.items() if not value]
if missing_keys:
    raise ValueError(f"Missing API keys: {', '.join(missing_keys)}. Please set them in your .env file.")

# 初始化AI客户端
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.siliconflow.cn/v1/")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# 请求模型
class WeatherRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    query: str = Field(..., min_length=1)
    forecast_days: Optional[int] = Field(5, ge=1, le=14)
    model_type: Optional[str] = Field("gemini", pattern="^(gemini|deepseek)$")

class FollowupRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    query: str = Field(..., min_length=1)
    forecast_days: Optional[int] = Field(5, ge=1, le=14)
    model_type: Optional[str] = Field("gemini", pattern="^(gemini|deepseek)$")

class CityRequest(BaseModel):
    city: str = Field(..., min_length=1)

def generate_cache_key(latitude: float, longitude: float, forecast_days: int, model_type: str) -> str:
    key_string = f"{latitude}_{longitude}_{forecast_days}_{model_type}"
    return hashlib.md5(key_string.encode()).hexdigest()

# 静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.get("/sw.js")
async def get_sw():
    return FileResponse("static/sw.js", media_type="application/javascript")

@app.post("/get_weather_advice")
async def get_weather_advice(request: Request, weather_request: WeatherRequest):
    try:
        # 添加提示语到查询
        enhanced_query = weather_request.query + '''
请以友好而专业的方式回答，将结构化信息融入自然对话中。

🎯 基本框架：
1. 开场白（简短友好）：
   - 问候语
   - 📍 位置信息：[经度，纬度] - [地理名称]
   - 天气概览：用一句话概括当前天气特点

2. 核心天气信息（分类呈现）：
   ⏰ 实时天气：
   温度：[数值]°C
   体感：[数值]°C
   湿度：[数值]%
   风向：[方向] [风速]m/s
   
   🌡️ 今日温差：
   最高温：[数值]°C
   最低温：[数值]°C
   温差：[数值]°C

   💨 空气状况：
   AQI指数：[数值]
   主要污染物：[名称]
   空气质量：[等级描述]

3. 📅 未来天气预报（按日期分段）：
   "YYYY-MM-DD"
   - 天气特征：[描述]
   - 温度区间：[最低温]-[最高温]°C
   - 降水概率：[数值]%
   - 关键提醒：[重点信息]
   ---

4. 👔 实用建议：
   > 穿衣建议：[具体建议]
   > 出行建议：[具体建议]
   > 活动建议：[具体建议]

5. ⚠️ 特别提醒（如有）：
   - 极端天气预警
   - 特殊天气注意事项
   - 健康防护建议

6. 结语：
   - 温馨提示或祝福语
   - 表达关心

请注意：
1. 在保持结构清晰的同时，用自然的语言连接各部分
2. 重要数据要用"引号"标注
3. 关键提醒使用 > 符号
4. 在合适的地方使用emoji增加可读性
5. 极端天气使用 ⚠️ 突出显示
6. 用分隔线(---)区分不同日期的预报

请用温和友好的语气输出，但确保信息完整且结构清晰。'''
        
        # 生成缓存键
        cache_key = generate_cache_key(
            weather_request.latitude,
            weather_request.longitude,
            weather_request.forecast_days,
            weather_request.model_type
        )

        # 确保天数在有效范围内
        forecast_days = max(1, min(weather_request.forecast_days, 14))

        async with aiohttp.ClientSession() as session:
            # 获取天气数据
            weather_url = f"https://my.meteoblue.com/packages/basic-3h?apikey={METEOBLUE_API_KEY}&lat={weather_request.latitude}&lon={weather_request.longitude}&format=json&windspeed=ms-1&forecast_days={forecast_days}"
            async with session.get(weather_url) as weather_response:
                weather_response.raise_for_status()
                weather_data = await weather_response.json()

            # 获取空气质量数据
            air_quality_url = f"https://api.waqi.info/feed/geo:{weather_request.latitude};{weather_request.longitude}/?token={WAQI_TOKEN}"
            async with session.get(air_quality_url) as air_quality_response:
                air_quality_response.raise_for_status()
                air_quality_data = await air_quality_response.json()

        # 准备LLM上下文
        context = f"""
        Weather Data (for {forecast_days} days): {json.dumps(weather_data)}
        Air Quality Data: {json.dumps(air_quality_data)}
        User Location: Latitude {weather_request.latitude}, Longitude {weather_request.longitude}
        """

        async def generate_stream() -> AsyncGenerator[str, None]:
            try:
                if weather_request.model_type == 'deepseek':
                    stream = deepseek_client.chat.completions.create(
                        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                        messages=[{"role": "user", "content": f"{context}\n\nUser Query: {enhanced_query}"}],
                        stream=True
                    )
                    content = ""
                    reasoning_content = ""
                    try:
                        for chunk in stream:
                            if await request.is_disconnected():
                                logger.info("Client disconnected, stopping stream")
                                return
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                yield f"data: {json.dumps({'content': content})}\n\n"
                            if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                                reasoning_content = chunk.choices[0].delta.reasoning_content
                                yield f"data: {json.dumps({'reasoning_content': reasoning_content})}\n\n"
                    except ConnectionResetError:
                        logger.warning("Connection reset by client")
                        return
                    except Exception as e:
                        logger.error(f"Stream processing error: {str(e)}")
                        raise
                elif weather_request.model_type == 'gemini':
                    stream = gemini_client.models.generate_content_stream(
                        model="gemini-2.0-flash-thinking-exp-01-21",
                        contents=f"{context}\n\nUser Query: {enhanced_query}"
                    )
                    started = False
                    try:
                        for chunk in stream:
                            if await request.is_disconnected():
                                logger.info("Client disconnected, stopping stream")
                                return
                            if not started:
                                started = True
                                yield f"data: {json.dumps({'start': True})}\n\n"
                            if chunk.text:
                                # 确保表格格式在 JSON 序列化时保持不变
                                formatted_text = chunk.text.replace('\\n', '\n').replace('\\t', '\t')
                                yield f"data: {json.dumps({'content': formatted_text, 'preserve_format': True})}\n\n"
                        if not await request.is_disconnected():
                            yield "data: {\"type\": \"done\"}\n\n"  # 修改完成信号的格式
                    except ConnectionResetError:
                        logger.warning("Connection reset by client")
                        return
                    except Exception as e:
                        logger.error(f"Stream processing error: {str(e)}")
                        raise
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported model type: {weather_request.model_type}")
                
                # 移除这里的 DONE 信号，因为已经在各自的处理分支中发送了
                # if not await request.is_disconnected():
                #     yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {str(e)}", exc_info=True)
                if not await request.is_disconnected():
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    except aiohttp.ClientError as e:
        raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error in get_weather_advice: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/ask_followup")
async def handle_followup_question(request: Request, followup_request: FollowupRequest):
    try:
        enhanced_query = followup_request.query + ' 请直接输出回答。保持原有格式风格，使用分割线和星级评价。不要使用任何Markdown语法标记。'
        
        cache_key = generate_cache_key(
            followup_request.latitude,
            followup_request.longitude,
            followup_request.forecast_days,
            followup_request.model_type
        )
        
        cached_data = weather_cache.get(cache_key)
        if not cached_data:
            raise HTTPException(status_code=400, detail="请先完成初始天气查询才能进行追问")

        async def generate_stream() -> AsyncGenerator[str, None]:
            try:
                if followup_request.model_type == 'deepseek':
                    messages = [{
                        "role": "user",
                        "content": f"""
                        天气数据：{json.dumps(cached_data['weather_data'])}
                        空气质量数据：{json.dumps(cached_data['air_quality_data'])}
                        用户位置：纬度 {followup_request.latitude}，经度 {followup_request.longitude}

                        用户追问：{enhanced_query}
                        """
                    }]
                    stream = deepseek_client.chat.completions.create(
                        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                        messages=messages,
                        stream=True
                    )
                    content = ""
                    reasoning_content = ""
                    try:
                        for chunk in stream:
                            if await request.is_disconnected():
                                logger.info("Client disconnected, stopping stream")
                                return
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                yield f"data: {json.dumps({'content': content})}\n\n"
                            if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                                reasoning_content = chunk.choices[0].delta.reasoning_content
                                yield f"data: {json.dumps({'reasoning_content': reasoning_content})}\n\n"
                    except ConnectionResetError:
                        logger.warning("Connection reset by client")
                        return
                    except Exception as e:
                        logger.error(f"Stream processing error: {str(e)}")
                        raise
                elif followup_request.model_type == 'gemini':
                    messages = f"""
                        天气数据：{json.dumps(cached_data['weather_data'])}
                        空气质量数据：{json.dumps(cached_data['air_quality_data'])}
                        用户位置：纬度 {followup_request.latitude}，经度 {followup_request.longitude}

                        用户追问：{enhanced_query}
                        """
                    stream = gemini_client.models.generate_content_stream(
                        model="gemini-2.0-flash-thinking-exp-01-21",
                        contents=messages
                    )
                    started = False
                    try:
                        for chunk in stream:
                            if await request.is_disconnected():
                                logger.info("Client disconnected, stopping stream")
                                return
                            if not started:
                                started = True
                                yield f"data: {json.dumps({'start': True})}\n\n"
                            if chunk.text:
                                # 确保表格格式在 JSON 序列化时保持不变
                                formatted_text = chunk.text.replace('\\n', '\n').replace('\\t', '\t')
                                yield f"data: {json.dumps({'content': formatted_text, 'preserve_format': True})}\n\n"
                        if not await request.is_disconnected():
                            yield "data: {\"type\": \"done\"}\n\n"  # 修改完成信号的格式
                    except ConnectionResetError:
                        logger.warning("Connection reset by client")
                        return
                    except Exception as e:
                        logger.error(f"Stream processing error: {str(e)}")
                        raise
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported model type: {followup_request.model_type}")
                
                # 移除这里的 DONE 信号，因为已经在各自的处理分支中发送了
                # if not await request.is_disconnected():
                #     yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {str(e)}", exc_info=True)
                if not await request.is_disconnected():
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Error in handle_followup_question: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理追问时出错: {str(e)}")

@app.post("/get_city_location")
async def get_city_location(request: CityRequest):
    if not request.city:
        raise HTTPException(status_code=400, detail="城市名称不能为空")

    try:
        # 构建提示词，要求Gemini返回精确的经纬度
        prompt = f"""
        请提供{request.city}的精确经纬度坐标。
        要求：
        1. 只返回经纬度数值，格式为：纬度,经度
        2. 如果是省份，返回省会城市的经纬度
        3. 如果是直辖市或特别行政区，返回其市中心经纬度
        4. 数值精确到小数点后4位
        5. 不要包含任何其他文字说明
        示例返回格式：39.9042,116.4074
        """

        # 使用Gemini API获取经纬度
        llm_response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt
        )

        # 解析响应
        coordinates = llm_response.text.strip().split(',')
        if len(coordinates) != 2:
            raise ValueError("Invalid coordinates format")

        latitude = float(coordinates[0])
        longitude = float(coordinates[1])

        # Validate coordinates are within reasonable global range
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError("Invalid coordinates range")

        return JSONResponse(content={
            'latitude': latitude,
            'longitude': longitude
        })

    except (ValueError, IndexError) as e:
        raise HTTPException(status_code=400, detail=f"无法解析城市位置信息: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取城市位置信息时出错: {str(e)}")

if __name__ == "__main__":
    # 生产环境配置
    uvicorn_config = uvicorn.Config(
        "app:app",
        host="0.0.0.0",
        port=8000,
        workers=4,  # 根据CPU核心数调整
        loop="uvloop",
        http="httptools",
        log_level="info",
        reload=True,  # 生产环境禁用热重
        access_log=True,
    )
    server = uvicorn.Server(uvicorn_config)
    server.run()
