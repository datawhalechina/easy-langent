import os
import json
import requests
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain.tools import tool
from pydantic import BaseModel, Field

# 1. 严谨加载环境变量
load_dotenv(override=True)

def _validate_env():
    """检查必要的环境变量是否存在"""
    required_keys = ["OPENWEATHER_API_KEY", "TAVILY_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        raise ValueError(f"CRITICAL ERROR: 缺少必要的环境变量: {missing_keys}。请检查 .env 文件。")

# 初始化时立即检查
_validate_env()

# --- 工具定义 ---

# TavilySearch 会自动读取环境变量中的 TAVILY_API_KEY
search_tool = TavilySearch(max_results=5, topic="general")

class WeatherQuery(BaseModel):
    loc: str = Field(description="The location name of the city")

@tool(args_schema=WeatherQuery)
def get_weather(loc):
    """
    查询即时天气函数
    :param loc: 必要参数，字符串类型，用于表示查询天气的具体城市名称，\
    注意，中国的城市需要用对应城市的英文名称代替，例如如果需要查询北京市天气，则loc参数需要输入'Beijing'；
    :return：OpenWeather API查询即时天气的结果，包含温度、天气状况等信息。
    """
    # 获取 Key (此前已校验过存在性)
    api_key = os.getenv("OPENWEATHER_API_KEY")
    
    # Step 1.构建请求
    url = "https://api.openweathermap.org/data/2.5/weather"

    # Step 2.设置查询参数
    params = {
        "q": loc,               
        "appid": api_key,
        "units": "metric",            
        "lang":"zh_cn"                
    }

    # Step 3.发送GET请求
    try:
        response = requests.get(url, params=params, timeout=10) # 增加 timeout 防止死锁
        response.raise_for_status()
        
        # Step 4.解析响应
        data = response.json()
        return json.dumps(data, ensure_ascii=False)
    except requests.exceptions.RequestException as e:
        return f"查询天气网络错误: {str(e)}"
    except Exception as e:
        return f"查询天气发生未知错误: {str(e)}"

def get_tools():
    return [search_tool, get_weather]