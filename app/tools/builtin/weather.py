"""
天气工具：双 Provider（QWeather 优先 + Open-Meteo 回退 + Stub 兜底）。

Provider 链（按配置依次尝试）：
    1. QWeather（和风天气）：需 QWEATHER_HOST + QWEATHER_API_KEY
       —— 城市解析用内置 location ID 表（避免依赖 geo/lookup，该接口在
          部分自定义 host 上不可用）；未知城市尝试 geo/lookup，失败回退下一级
    2. Open-Meteo：免费无需 key（内置经纬度映射）
    3. Stub 数据：完全离线（教学 / 测试）

设计：任一 Provider 失败都回退下一个，保证 Agent 永不因天气 API 故障中断。
"""
import httpx
from pydantic import BaseModel, Field

from app.config import Settings
from app.errors import ToolExecutionError

# 内置城市 location ID（QWeather 用，和风天气标准城市 ID）
_CITY_LOCATION_IDS: dict[str, str] = {
    "北京": "101010100",
    "上海": "101020100",
    "广州": "101280101",
    "深圳": "101280601",
    "杭州": "101210101",
    "成都": "101270101",
    "武汉": "101200101",
    "西安": "101110101",
    "南京": "101190101",
    "重庆": "101040100",
}

# 内置城市经纬度（Open-Meteo 用）
_CITY_COORDS: dict[str, tuple[float, float]] = {
    "北京": (39.9042, 116.4074),
    "上海": (31.2304, 121.4737),
    "广州": (23.1291, 113.2644),
    "深圳": (22.5431, 114.0579),
    "杭州": (30.2741, 120.1551),
    "成都": (30.5728, 104.0668),
    "武汉": (30.5928, 114.3055),
    "西安": (34.3416, 108.9398),
    "南京": (32.0603, 118.7969),
    "重庆": (29.5630, 106.5516),
}

# Stub 天气数据（最后兜底）
_STUB_WEATHER: dict[str, str] = {
    "北京": "晴，25°C，微风",
    "上海": "多云，22°C，东南风3级",
    "广州": "小雨，28°C，湿度85%",
    "深圳": "晴，29°C，适宜户外活动",
    "杭州": "阴，21°C，局部有雾",
    "成都": "晴，24°C，空气优",
}


class WeatherArgs(BaseModel):
    """天气查询参数。"""

    city: str = Field(description="城市名（中文，如 北京）")


# ---------------------------------------------------------------------------
# Provider 1: QWeather（和风天气）
# ---------------------------------------------------------------------------
def _qweather_location_id(city: str, settings: Settings) -> str | None:
    """解析城市 location ID：先查内置表，未命中再试 geo/lookup。"""
    if city in _CITY_LOCATION_IDS:
        return _CITY_LOCATION_IDS[city]
    try:
        r = httpx.get(
            f"https://{settings.qweather_host}/v7/geo/lookup",
            params={"location": city, "number": 1},
            headers={"X-QW-Api-Key": settings.qweather_api_key},
            timeout=settings.qweather_timeout_seconds,
        )
        if r.status_code == 200:
            locations = r.json().get("location") or []
            if locations:
                return locations[0]["id"]
    except (httpx.HTTPError, ValueError):
        pass
    return None


def _fetch_qweather(city: str, settings: Settings) -> str:
    """QWeather 实时天气：内置 location ID 优先，lookup 兜底。"""
    if not settings.qweather_api_key or not settings.qweather_host:
        raise ToolExecutionError("QWeather 未配置（QWEATHER_HOST / QWEATHER_API_KEY）")

    location_id = _qweather_location_id(city, settings)
    if not location_id:
        raise ToolExecutionError(f"QWeather 无法解析城市「{city}」")

    headers = {"X-QW-Api-Key": settings.qweather_api_key}
    base = f"https://{settings.qweather_host}"
    try:
        now = httpx.get(
            f"{base}/v7/weather/now",
            params={"location": location_id},
            headers=headers,
            timeout=settings.qweather_timeout_seconds,
        )
    except httpx.HTTPError as exc:
        raise ToolExecutionError(f"QWeather 天气请求失败: {exc}", transient=True) from exc
    if now.status_code == 401:
        raise ToolExecutionError("QWeather 鉴权失败：请检查 QWEATHER_API_KEY / QWEATHER_HOST")
    if now.status_code != 200:
        raise ToolExecutionError(f"QWeather 天气返回 {now.status_code}")

    try:
        data = now.json().get("now") or {}
    except ValueError as exc:
        raise ToolExecutionError(f"QWeather 天气响应非法: {exc}") from exc
    if not data:
        raise ToolExecutionError(f"QWeather 未返回「{city}」天气数据")

    text = data.get("text") or ""
    temp = data.get("temp") or ""
    humidity = data.get("humidity") or ""
    wind_dir = data.get("windDir") or ""
    wind_scale = data.get("windScale") or ""
    parts = [f"{city}：{text}"]
    if temp:
        parts.append(f"{temp}°C")
    if humidity:
        parts.append(f"湿度{humidity}%")
    if wind_dir and wind_scale:
        # QWeather 的 windDir 已含"风"字（如"南风"），避免"南风风2级"
        dir_clean = wind_dir[:-1] if wind_dir.endswith("风") else wind_dir
        parts.append(f"{dir_clean}风{wind_scale}级")
    return "，".join(parts)


# ---------------------------------------------------------------------------
# Provider 2: Open-Meteo（免费，无需 key）
# ---------------------------------------------------------------------------
def _fetch_open_meteo(city: str, lat: float, lon: float, settings: Settings) -> str:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m",
        "timezone": "Asia/Shanghai",
    }
    try:
        resp = httpx.get(url, params=params, timeout=settings.weather_api_timeout_seconds)
    except httpx.HTTPError as exc:
        raise ToolExecutionError(f"天气服务请求失败: {exc}", transient=True) from exc
    if resp.status_code != 200:
        raise ToolExecutionError(f"天气服务返回 {resp.status_code}")
    try:
        data = resp.json()
        cur = data.get("current", {})
    except ValueError as exc:
        raise ToolExecutionError(f"天气响应格式非法: {exc}") from exc
    temp = cur.get("temperature_2m")
    if temp is None:
        raise ToolExecutionError(f"未获取到 {city} 的天气数据")
    desc = _wmo_description(cur.get("weather_code"))
    wind = cur.get("wind_speed_10m")
    humidity = cur.get("relative_humidity_2m")
    return f"{city}：{desc}，{temp}°C，湿度{humidity}%，风速{wind}km/h"


def _wmo_description(code: int | None) -> str:
    table = {
        0: "晴", 1: "大部晴朗", 2: "多云", 3: "阴",
        45: "雾", 48: "雾凇", 51: "毛毛雨", 53: "小毛毛雨", 55: "大毛毛雨",
        61: "小雨", 63: "中雨", 65: "大雨",
        71: "小雪", 73: "中雪", 75: "大雪",
        80: "阵雨", 81: "强阵雨", 82: "暴雨",
        95: "雷暴", 96: "雷暴伴冰雹", 99: "强雷暴伴冰雹",
    }
    return table.get(code, "未知")


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
def weather_handler(city: str) -> str:
    """查询城市天气；Provider 链依次尝试，全部失败回退 Stub。"""
    city = city.strip()
    settings = Settings()

    # 1) QWeather（配置了才尝试）
    if settings.qweather_api_key and settings.qweather_host:
        try:
            return _fetch_qweather(city, settings)
        except ToolExecutionError:
            pass  # 回退下一级

    # 2) Open-Meteo（城市有经纬度）
    if settings.weather_use_real_api and city in _CITY_COORDS:
        lat, lon = _CITY_COORDS[city]
        try:
            return _fetch_open_meteo(city, lat, lon, settings)
        except ToolExecutionError:
            pass  # 回退 Stub

    # 3) Stub 兜底
    if city in _STUB_WEATHER:
        return _STUB_WEATHER[city]
    supported = "、".join(_STUB_WEATHER)
    raise ToolExecutionError(f"未找到城市「{city}」的天气数据，支持：{supported}")
