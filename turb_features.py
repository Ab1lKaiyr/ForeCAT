"""
ForeCAT-KZ — физика турбулентности ясного неба (CAT).

Этот файл общий для блокнота обучения и для приложения:
  1) скачивает прогноз ветра и температуры на высотах полёта (Open-Meteo, модель GFS);
  2) считает физические признаки турбулентности:
       - сдвиг ветра по высоте (VWS),
       - устойчивость воздуха (N^2) и число Ричардсона (Ri),
       - деформацию, сходимость и вихрь потока,
       - индексы Эллрода TI1 и TI2 (их используют авиаметеорологи);
  3) даёт простое физическое правило «светофор» без ИИ.
"""

import math
import time

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------

# Уровни давления (гПа) примерно на высоте полёта: 300 ≈ 9,2 км, 250 ≈ 10,4 км, 200 ≈ 11,8 км
LEVELS = (200, 250, 300)
# Слои между уровнями: (верх, низ)
LAYERS = ((200, 250), (250, 300))

# Насколько градусов отступаем от точки на север/юг/восток/запад,
# чтобы посчитать, как ветер меняется по горизонтали
STENCIL_DEG = 0.5

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORY_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Глобальная модель GFS (NOAA). У Open-Meteo менялись названия моделей,
# поэтому пробуем несколько вариантов по очереди.
MODEL_CANDIDATES = ("gfs_global", "ncep_gfs_global_025", "gfs_seamless", "ncep_gfs_seamless")
_working_model = None

G = 9.81          # ускорение свободного падения, м/с²
R_EARTH = 6371000  # радиус Земли, м

# Признаки, которые получает модель ИИ (в удобных для чтения единицах)
FEATURES = ["ws_mid", "vws", "n2", "ri", "deform", "converg", "vort", "ti1", "ti2"]

FEATURE_NAMES_RU = {
    "ws_mid": "Скорость ветра, м/с",
    "vws": "Сдвиг ветра по высоте, узлы на 1000 футов",
    "n2": "Устойчивость N², 10⁻⁴ с⁻²",
    "ri": "Число Ричардсона Ri",
    "deform": "Деформация потока, 10⁻⁵ с⁻¹",
    "converg": "Сходимость потока, 10⁻⁵ с⁻¹",
    "vort": "Вихрь (завихрённость), 10⁻⁵ с⁻¹",
    "ti1": "Индекс Эллрода TI1, 10⁻⁷ с⁻²",
    "ti2": "Индекс Эллрода TI2, 10⁻⁷ с⁻²",
}

RISK_NAMES = {0: "спокойно", 1: "возможна болтанка", 2: "сильная болтанка вероятна"}
RISK_COLORS = {0: "#2e9e5b", 1: "#e0a100", 2: "#d64545"}


# ---------------------------------------------------------------------------
# Высота полёта и слои
# ---------------------------------------------------------------------------

def fl_to_pressure(flight_level_ft):
    """Высота в футах (например 35000) → давление в гПа по стандартной атмосфере."""
    h = float(flight_level_ft) * 0.3048
    if h <= 11000:
        return 1013.25 * (1 - 2.25577e-5 * h) ** 5.25588
    return 226.32 * math.exp(-(h - 11000) / 6341.62)


def choose_layer(pressure_hpa):
    """Выбирает слой (верх, низ), внутри которого летит самолёт."""
    if pressure_hpa < 250:
        return LAYERS[0]
    return LAYERS[1]


def hourly_variables(levels=LEVELS):
    names = []
    for p in levels:
        names += [
            f"wind_speed_{p}hPa",
            f"wind_direction_{p}hPa",
            f"temperature_{p}hPa",
            f"geopotential_height_{p}hPa",
        ]
    return names


def stencil(lat, lon, d=STENCIL_DEG):
    """5 точек «крестиком»: центр, север, юг, восток, запад."""
    return [
        (lat, lon),
        (lat + d, lon),
        (lat - d, lon),
        (lat, lon + d),
        (lat, lon - d),
    ]


# ---------------------------------------------------------------------------
# Загрузка данных Open-Meteo
# ---------------------------------------------------------------------------

def _request_json(url, params, retries=5):
    for attempt in range(retries):
        r = requests.get(url, params=params, timeout=90)
        if r.status_code == 429:
            # Превышен бесплатный лимит в минуту — ждём и пробуем ещё раз
            wait = 60 * (attempt + 1)
            print(f"  Open-Meteo просит подождать (лимит запросов). Жду {wait} с…")
            time.sleep(wait)
            continue
        try:
            data = r.json()
        except ValueError:
            r.raise_for_status()
            raise
        return r.status_code, data
    raise RuntimeError(
        "Open-Meteo не отвечает из-за лимита. Возможно, дневной лимит исчерпан — "
        "продолжите завтра: уже скачанные данные сохранены в кэше."
    )


def fetch_openmeteo(points, start_date=None, end_date=None, forecast_days=None,
                    levels=LEVELS):
    """
    Скачивает почасовые данные для списка точек [(lat, lon), ...].
      - для обучения: start_date/end_date ('2025-01-15') — архив прогнозов;
      - для приложения: forecast_days=3 — свежий прогноз.
    Возвращает список таблиц pandas (по одной на точку), строки — часы (UTC).
    """
    global _working_model
    historical = start_date is not None
    url = HISTORY_URL if historical else FORECAST_URL
    params = {
        "latitude": ",".join(f"{la:.4f}" for la, _ in points),
        "longitude": ",".join(f"{lo:.4f}" for _, lo in points),
        "hourly": ",".join(hourly_variables(levels)),
        "wind_speed_unit": "ms",
        "timezone": "GMT",
    }
    if historical:
        params["start_date"] = start_date
        params["end_date"] = end_date or start_date
    else:
        params["forecast_days"] = forecast_days or 3

    models = [_working_model] if _working_model else list(MODEL_CANDIDATES)
    last_error = None
    for model in models:
        params["models"] = model
        status, data = _request_json(url, params)
        if isinstance(data, dict) and data.get("error"):
            last_error = data.get("reason", "неизвестная ошибка")
            continue
        _working_model = model
        items = data if isinstance(data, list) else [data]
        frames = []
        for item in items:
            hourly = item["hourly"]
            df = pd.DataFrame(hourly)
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time").apply(pd.to_numeric, errors="coerce")
            frames.append(df)
        return frames
    raise RuntimeError(f"Open-Meteo вернул ошибку: {last_error}")


# ---------------------------------------------------------------------------
# Физика
# ---------------------------------------------------------------------------

def wind_uv(speed, direction_deg):
    """Скорость и направление ветра → компоненты u (на восток) и v (на север).
    Метеорологи указывают, ОТКУДА дует ветер, поэтому стоят минусы."""
    rad = np.deg2rad(direction_deg)
    return -speed * np.sin(rad), -speed * np.cos(rad)


def potential_temperature(temp_c, pressure_hpa):
    """Потенциальная температура θ: какой была бы температура воздуха,
    если его опустить до давления 1000 гПа. Нужна, чтобы оценить устойчивость."""
    return (temp_c + 273.15) * (1000.0 / pressure_hpa) ** 0.2857


def _level(df, p):
    u, v = wind_uv(df[f"wind_speed_{p}hPa"], df[f"wind_direction_{p}hPa"])
    return {
        "u": u,
        "v": v,
        "theta": potential_temperature(df[f"temperature_{p}hPa"], p),
        "z": df[f"geopotential_height_{p}hPa"],
    }


def compute_features(frames5, lat, layer, d=STENCIL_DEG):
    """
    frames5 — 5 таблиц (центр, север, юг, восток, запад) из fetch_openmeteo.
    layer   — слой (верх, низ), например (200, 250).
    Возвращает таблицу признаков по часам.
    """
    p_top, p_bot = layer
    c, n, s, e, w = frames5

    # --- Вертикаль: сдвиг ветра и устойчивость (в центральной точке) ---
    top, bot = _level(c, p_top), _level(c, p_bot)
    dz = (top["z"] - bot["z"]).where(lambda x: x > 100)          # толщина слоя, м
    vws = np.sqrt((top["u"] - bot["u"]) ** 2 + (top["v"] - bot["v"]) ** 2) / dz  # 1/с
    theta_mean = (top["theta"] + bot["theta"]) / 2
    n2 = G / theta_mean * (top["theta"] - bot["theta"]) / dz     # 1/с²
    ri = n2 / np.maximum(vws, 1e-6) ** 2
    ri = ri.clip(-5, 50)

    # --- Горизонталь: ветер посередине слоя в 4 соседних точках ---
    def mid_uv(df):
        a, b = _level(df, p_top), _level(df, p_bot)
        return (a["u"] + b["u"]) / 2, (a["v"] + b["v"]) / 2

    u_c, v_c = mid_uv(c)
    u_n, v_n = mid_uv(n)
    u_s, v_s = mid_uv(s)
    u_e, v_e = mid_uv(e)
    u_w, v_w = mid_uv(w)

    dy = 2 * math.radians(d) * R_EARTH
    dx = 2 * math.radians(d) * R_EARTH * math.cos(math.radians(lat))

    du_dx, dv_dx = (u_e - u_w) / dx, (v_e - v_w) / dx
    du_dy, dv_dy = (u_n - u_s) / dy, (v_n - v_s) / dy

    stretch = du_dx - dv_dy                    # растяжение
    shear_h = dv_dx + du_dy                    # горизонтальный сдвиг
    deform = np.sqrt(stretch ** 2 + shear_h ** 2)
    converg = -(du_dx + dv_dy)                 # сходимость (воздух «сжимается»)
    vort = dv_dx - du_dy                       # вихрь (закрутка)

    ti1 = vws * deform                         # индекс Эллрода 1
    ti2 = vws * (deform + converg)             # индекс Эллрода 2

    out = pd.DataFrame({
        "ws_mid": np.sqrt(u_c ** 2 + v_c ** 2),
        "vws": vws * 1.94384 * 304.8,          # 1/с → узлы на 1000 футов
        "n2": n2 * 1e4,
        "ri": ri,
        "deform": deform * 1e5,
        "converg": converg * 1e5,
        "vort": vort * 1e5,
        "ti1": ti1 * 1e7,
        "ti2": ti2 * 1e7,
    })
    return out


# ---------------------------------------------------------------------------
# Простое физическое правило («светофор») — без ИИ
# ---------------------------------------------------------------------------

def rule_risk(features):
    """
    Уровень риска 0/1/2 по классическим ориентирам:
      - индекс Эллрода TI1 ≥ 4 → возможна болтанка, ≥ 8 → сильная;
      - число Ричардсона Ri < 1 → возможна, < 0,5 → сильная.
    Берём худший из двух.
    """
    ti1 = features["ti1"]
    ri = features["ri"]
    lvl_ti1 = np.select([ti1 >= 8, ti1 >= 4], [2, 1], 0)
    lvl_ri = np.select([ri < 0.5, ri < 1.0], [2, 1], 0)
    return pd.Series(np.maximum(lvl_ti1, lvl_ri), index=features.index)


def probability_to_risk(p):
    """Вероятность от модели ИИ → уровень светофора."""
    p = np.asarray(p)
    return np.select([p >= 0.65, p >= 0.4], [2, 1], 0)


# ---------------------------------------------------------------------------
# Маршрут
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH / 1000 * math.asin(math.sqrt(a))


def route_points(lat1, lon1, lat2, lon2, step_km=100, trim=0.08):
    """Точки вдоль кратчайшего пути (по дуге большого круга).
    Начало и конец маршрута (набор высоты и снижение) отрезаем."""
    total = haversine_km(lat1, lon1, lat2, lon2)
    n = int(np.clip(round(total / step_km), 4, 25))
    p1, l1, p2, l2 = map(math.radians, (lat1, lon1, lat2, lon2))
    a = np.array([math.cos(p1) * math.cos(l1), math.cos(p1) * math.sin(l1), math.sin(p1)])
    b = np.array([math.cos(p2) * math.cos(l2), math.cos(p2) * math.sin(l2), math.sin(p2)])
    omega = math.acos(float(np.clip(a @ b, -1, 1)))
    pts = []
    for f in np.linspace(trim, 1 - trim, n):
        if omega < 1e-9:
            v = a
        else:
            v = (math.sin((1 - f) * omega) * a + math.sin(f * omega) * b) / math.sin(omega)
        lat = math.degrees(math.atan2(v[2], math.hypot(v[0], v[1])))
        lon = math.degrees(math.atan2(v[1], v[0]))
        pts.append((lat, lon, f * total))
    return pts, total
