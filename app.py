# backend.py
from flask import Flask, request, jsonify, send_from_directory
import requests
import json
from openai import OpenAI
import os
import urllib.parse
import hashlib
from dotenv import load_dotenv
from flask_cors import CORS
import time
from google import genai  # Import Gemini API library
import logging
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化天气缓存字典
weather_cache = {}

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)
CORS(app)

METEOBLUE_API_KEY = os.getenv("METEOBLUE_API_KEY")
WAQI_TOKEN = os.getenv("WAQI_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
BAIDU_MAP_AK = os.getenv("BAIDU_MAP_AK")      # Baidu Maps AK
BAIDU_MAP_SK = os.getenv("BAIDU_MAP_SK")      # Baidu Maps SK
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Gemini API Key

# Simplify API key validation
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

deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com") # Initialize DeepSeek client
gemini_client = genai.Client(api_key=GEMINI_API_KEY) # Initialize Gemini client

def generate_cache_key(latitude, longitude, forecast_days, model_type):
    key_string = f"{latitude}_{longitude}_{forecast_days}_{model_type}"
    return hashlib.md5(key_string.encode()).hexdigest()

def handle_api_errors(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            return jsonify({'error': f"API request failed: {str(e)}"}), 500
        except Exception as e:
            return jsonify({'error': f"An error occurred: {str(e)}"}), 500
    return decorated_function

@app.route('/get_weather_advice', methods=['POST'])
def get_weather_advice():
    data = request.get_json()
    latitude = data['latitude']
    longitude = data['longitude']
    query = data['query'] + '请直接输出回答。同时请在最开头得出我的位置以及我的位置大致的名称，如果是询问优劣、最适合类型的问题，请使用五★进行评级。必须高度充分利用Markdown语法进行结构化，日期明显易读，日期与日期之间必修使用Markdown语法分割线。必须在回答文本中使用对应emoji以提升易读性'
    forecast_days = data.get('forecast_days', 5)  # Default to 5 if not provided
    model_type = data.get('model_type', 'gemini') # Default to gemini if not provided

    # 生成基于位置和天数的缓存键（不再包含时间因素）
    cache_key = generate_cache_key(latitude, longitude, forecast_days, model_type)

    # 确保天数在API有效范围内
    forecast_days = max(1, min(forecast_days, 14))

    # Fetch Weather API data
    weather_url = f"https://my.meteoblue.com/packages/basic-3h?apikey={METEOBLUE_API_KEY}&lat={latitude}&lon={longitude}&format=json&windspeed=ms-1&forecast_days={forecast_days}"
    try:
        weather_response = requests.get(weather_url)
        weather_response.raise_for_status()
        weather_data = weather_response.json()
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f"Error fetching weather data: {e}"}), 500

    # Fetch Air Quality API data
    air_quality_url = f"https://api.waqi.info/feed/geo:{latitude};{longitude}/?token={WAQI_TOKEN}"
    try:
        air_quality_response = requests.get(air_quality_url)
        air_quality_response.raise_for_status()
        air_quality_data = air_quality_response.json()
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f"Error fetching air quality data: {e}"}), 500

    # Prepare data for the LLM
    context = f"""
    Weather Data (for {forecast_days} days): {json.dumps(weather_data)}
    Air Quality Data: {json.dumps(air_quality_data)}
    User Location: Latitude {latitude}, Longitude {longitude}
    """

    # Select LLM client based on model_type
    if model_type == 'deepseek':
        llm_client = deepseek_client
        model_name = "deepseek-reasoner"
        messages = [{"role": "user", "content": f"{context}\n\nUser Query: {query}"}]
    elif model_type == 'gemini':
        llm_client = gemini_client
        model_name = "gemini-2.0-flash-thinking-exp-01-21"
        messages = f"{context}\n\nUser Query: {query}" # Gemini API expects a single string for content
    else:
        return jsonify({'error': f"Unsupported model type: {model_type}"}), 400

    # Send to LLM
    try:
        if model_type == 'deepseek':
            llm_response = llm_client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            ai_response = llm_response.choices[0].message.content
        elif model_type == 'gemini':
            llm_response = llm_client.models.generate_content(
                model=model_name,
                contents=messages
            )
            ai_response = llm_response.text
        else:
            return jsonify({'error': f"Unsupported model type: {model_type}"}), 400

    except Exception as e:
        return jsonify({'error': f"Error interacting with the LLM ({model_type}): {e}"}), 500

    # 将结果存入缓存（已移除时间戳）
    weather_cache[cache_key] = {
        'weather_data': weather_data,
        'air_quality_data': air_quality_data,
        'ai_response': ai_response
    }

    return jsonify({'ai_response': ai_response})


# Route to serve the index.html file (your frontend)
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/sw.js')
def sw():
    return send_from_directory('static', 'sw.js', mimetype='application/javascript')


# Optionally serve other static files (like CSS, JS, images if you had them separately)
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


@app.route('/ask_followup', methods=['POST'])
def handle_followup_question():
    data = request.get_json()
    latitude = data['latitude']
    longitude = data['longitude']
    forecast_days = data.get('forecast_days', 5)
    new_query = data['query'] + ' 请直接输出回答。保持原有格式风格，使用分割线和星级评价。'
    model_type = data.get('model_type', 'gemini') # Get model_type from request, default to gemini

    cache_key = generate_cache_key(latitude, longitude, forecast_days, model_type)
    cached_data = weather_cache.get(cache_key)

    # 强制要求必须存在缓存才能进行追问
    if not cached_data:
        return jsonify({'error': '请先完成初始天气查询才能进行追问'}), 400

    # Select LLM client based on model_type
    if model_type == 'deepseek':
        llm_client = deepseek_client
        model_name = "deepseek-reasoner"
        messages = [{
            "role": "user",
            "content": f"""
            天气数据：{json.dumps(cached_data['weather_data'])}
            空气质量数据：{json.dumps(cached_data['air_quality_data'])}
            用户位置：纬度 {latitude}，经度 {longitude}

            用户追问：{new_query}
            """
        }]
    elif model_type == 'gemini':
        llm_client = gemini_client
        model_name = "gemini-2.0-flash"
        messages = f"""
            天气数据：{json.dumps(cached_data['weather_data'])}
            空气质量数据：{json.dumps(cached_data['air_quality_data'])}
            用户位置：纬度 {latitude}，经度 {longitude}

            用户追问：{new_query}
            """
    else:
        return jsonify({'error': f"Unsupported model type: {model_type}"}), 400


    try:
        if model_type == 'deepseek':
            llm_response = llm_client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            ai_response = llm_response.choices[0].message.content
        elif model_type == 'gemini':
            llm_response = llm_client.models.generate_content(
                model=model_name,
                contents=messages
            )
            ai_response = llm_response.text
        else:
            return jsonify({'error': f"Unsupported model type: {model_type}"}), 400
    except Exception as e:
        return jsonify({'error': f"处理追问时出错 ({model_type}): {e}"}), 500

    return jsonify({'ai_response': ai_response})

def get_client_ip():
    """Get client IP address from request headers."""
    headers_to_check = ['X-Forwarded-For', 'X-Real-IP']
    for header in headers_to_check:
        if header in request.headers:
            return request.headers[header].split(',')[0].strip()
    return request.remote_addr


@app.route('/get_ip_location', methods=['GET'])
def get_ip_location():
    client_ip = get_client_ip() # Use the function to get client IP

    print(f"Client IP (from get_client_ip): {client_ip}") # Debug log - Print the IP obtained

    effective_ip_for_baidu = client_ip # No more localhost logic

    # Baidu Maps API configuration
    host = "https://api.map.baidu.com"
    uri = "/location/ip"
    ak = BAIDU_MAP_AK
    sk = BAIDU_MAP_SK
    coor = "bd09ll" # Optional, for Baidu coordinates

    print(f"Using IP address for Baidu Maps API: {effective_ip_for_baidu}") # Debug log - print the IP used for API call

    params = {
        "ip": effective_ip_for_baidu,
        "coor": coor,
        "ak": ak,
    }

    # Calculate SN for Baidu Maps API
    params_arr = []
    for key in params:
        params_arr.append(key + "=" + params[key])
    query_str_no_sn = uri + "?" + "&".join(params_arr)
    encoded_str = urllib.parse.quote(query_str_no_sn, safe="/:=&?#+!$,;'@()*[]")
    raw_str = encoded_str + sk
    sn = hashlib.md5(urllib.parse.quote_plus(raw_str).encode("utf8")).hexdigest()
    query_str = query_str_no_sn + "&sn=" + sn
    url = host + query_str

    try:
        response = requests.get(url)
        response.raise_for_status()
        baidu_data = response.json()
        print("Baidu Maps API Response:") # Debug log
        print(json.dumps(baidu_data, indent=4, ensure_ascii=False)) # Debug log - print full API response

        if baidu_data.get('status') == 0: # Check if the API call was successful
            location_info = baidu_data.get('content', {}).get('address_detail', {})
            latitude = baidu_data.get('content', {}).get('point', {}).get('y') # Latitude is y in Baidu API
            longitude = baidu_data.get('content', {}).get('point', {}).get('x') # Longitude is x in Baidu API
            city = location_info.get('city', 'Unknown City')
            province = location_info.get('province', 'Unknown Province')
            return jsonify({'latitude': latitude, 'longitude': longitude, 'city': city, 'province': province})
        else:
            return jsonify({'error': f"Baidu Maps API error: {baidu_data.get('message', 'Unknown error')}"}), 500
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f"Error fetching IP location from Baidu Maps: {e}"}), 500


@app.route('/get_city_location', methods=['POST'])
def get_city_location():
    data = request.get_json()
    city = data.get('city')
    
    if not city:
        return jsonify({'error': '城市名称不能为空'}), 400

    try:
        # 构建提示词，要求Gemini返回精确的经纬度
        prompt = f"""
        请提供{city}的精确经纬度坐标。
        要求：
        1. 只返回经纬度数值，格式为：纬度,经度
        2. 如果是省份，返回省会城市的经纬度
        3. 如果是直辖市或特别行政区，返回其市中心经纬度
        4. 数值精确到小数点后4位
        5. 不要包含任何其他文字说明
        示例返回格式：39.9042,116.4074
        """

        # 使用Gemini API获取经纬度
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-lite-preview-02-05",
            contents=prompt
        )

        # 解析响应
        coordinates = response.text.strip().split(',')
        if len(coordinates) != 2:
            raise ValueError("Invalid coordinates format")

        latitude = float(coordinates[0])
        longitude = float(coordinates[1])

        # Validate coordinates are within reasonable global range
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError("Invalid coordinates range")

        return jsonify({
            'latitude': latitude,
            'longitude': longitude
        })

    except (ValueError, IndexError) as e:
        return jsonify({'error': f'无法解析城市位置信息: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'获取城市位置信息时出错: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')