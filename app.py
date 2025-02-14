# backend.py
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any
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

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化天气缓存字典
weather_cache: Dict[str, Any] = {}

# 加载环境变量
load_dotenv()

# 创建FastAPI应用
app = FastAPI(title="Weather AI Assistant")

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
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# 请求模型
class WeatherRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    latitude: float
    longitude: float
    query: str
    forecast_days: Optional[int] = 5
    model_type: Optional[str] = "gemini"

class FollowupRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    latitude: float
    longitude: float
    query: str
    forecast_days: Optional[int] = 5
    model_type: Optional[str] = "gemini"

class CityRequest(BaseModel):
    city: str

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
async def get_weather_advice(request: WeatherRequest):
    try:
        # 添加提示语到查询
        enhanced_query = request.query + '请直接输出回答。同时请在最开头得出我的位置以及我的位置大致的名称，如果是询问优劣、最适合类型的问题，请使用五★进行评级。必须高度充分利用Markdown语法进行结构化，日期明显易读，日期与日期之间必修使用Markdown语法分割线。必须在回答文本中使用对应emoji以提升易读性'
        
        # 生成缓存键
        cache_key = generate_cache_key(
            request.latitude,
            request.longitude,
            request.forecast_days,
            request.model_type
        )

        # 确保天数在有效范围内
        forecast_days = max(1, min(request.forecast_days, 14))

        async with aiohttp.ClientSession() as session:
            # 获取天气数据
            weather_url = f"https://my.meteoblue.com/packages/basic-3h?apikey={METEOBLUE_API_KEY}&lat={request.latitude}&lon={request.longitude}&format=json&windspeed=ms-1&forecast_days={forecast_days}"
            async with session.get(weather_url) as weather_response:
                weather_response.raise_for_status()
                weather_data = await weather_response.json()

            # 获取空气质量数据
            air_quality_url = f"https://api.waqi.info/feed/geo:{request.latitude};{request.longitude}/?token={WAQI_TOKEN}"
            async with session.get(air_quality_url) as air_quality_response:
                air_quality_response.raise_for_status()
                air_quality_data = await air_quality_response.json()

        # 准备LLM上下文
        context = f"""
        Weather Data (for {forecast_days} days): {json.dumps(weather_data)}
        Air Quality Data: {json.dumps(air_quality_data)}
        User Location: Latitude {request.latitude}, Longitude {request.longitude}
        """

        # 根据模型类型选择LLM客户端
        if request.model_type == 'deepseek':
            llm_response = deepseek_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": f"{context}\n\nUser Query: {enhanced_query}"}]
            )
            ai_response = llm_response.choices[0].message.content
        elif request.model_type == 'gemini':
            llm_response = gemini_client.models.generate_content(
                model="gemini-2.0-flash-thinking-exp-01-21",
                contents=f"{context}\n\nUser Query: {enhanced_query}"
            )
            ai_response = llm_response.text
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {request.model_type}")

        # 更新缓存
        weather_cache[cache_key] = {
            'weather_data': weather_data,
            'air_quality_data': air_quality_data,
            'ai_response': ai_response
        }

        return JSONResponse(content={'ai_response': ai_response})

    except aiohttp.ClientError as e:
        raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error in get_weather_advice: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/ask_followup")
async def handle_followup_question(request: FollowupRequest):
    try:
        enhanced_query = request.query + ' 请直接输出回答。保持原有格式风格，使用分割线和星级评价。'
        
        cache_key = generate_cache_key(
            request.latitude,
            request.longitude,
            request.forecast_days,
            request.model_type
        )
        
        cached_data = weather_cache.get(cache_key)
        if not cached_data:
            raise HTTPException(status_code=400, detail="请先完成初始天气查询才能进行追问")

        if request.model_type == 'deepseek':
            messages = [{
                "role": "user",
                "content": f"""
                天气数据：{json.dumps(cached_data['weather_data'])}
                空气质量数据：{json.dumps(cached_data['air_quality_data'])}
                用户位置：纬度 {request.latitude}，经度 {request.longitude}

                用户追问：{enhanced_query}
                """
            }]
            llm_response = deepseek_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=messages
            )
            ai_response = llm_response.choices[0].message.content
        elif request.model_type == 'gemini':
            messages = f"""
                天气数据：{json.dumps(cached_data['weather_data'])}
                空气质量数据：{json.dumps(cached_data['air_quality_data'])}
                用户位置：纬度 {request.latitude}，经度 {request.longitude}

                用户追问：{enhanced_query}
                """
            llm_response = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=messages
            )
            ai_response = llm_response.text
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {request.model_type}")

        return JSONResponse(content={'ai_response': ai_response})

    except Exception as e:
        logger.error(f"Error in handle_followup_question: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理追问时出错: {str(e)}")

@app.get("/get_ip_location")
async def get_ip_location(request: Request):
    def get_client_ip(request: Request) -> str:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "127.0.0.1"

    try:
        client_ip = get_client_ip(request)
        logger.info(f"Client IP: {client_ip}")

        # Baidu Maps API configuration
        host = "https://api.map.baidu.com"
        uri = "/location/ip"
        ak = BAIDU_MAP_AK
        
        # Make request to Baidu API
        url = f"{host}{uri}?ak={ak}&ip={client_ip}&coor=bd09ll"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                location_data = await response.json()

        return JSONResponse(content=location_data)

    except Exception as e:
        logger.error(f"Error in get_ip_location: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting location: {str(e)}")

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
            model="gemini-2.0-flash-lite-preview-02-05",
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)