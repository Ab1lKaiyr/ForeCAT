"""
ForeCAT-KZ — страница «Прогноз рейса»: будет ли трясти в полёте?

Запускается из app.py:   python -m streamlit run app.py
Модель ИИ (необязательно): положите model/forecat_model.joblib и model/model_info.json
из блокнота обучения. Без модели страница работает по физическому правилу.
"""

import json
import os
from datetime import timedelta, timezone

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

import turb_features as tf

KZ_TZ = timezone(timedelta(hours=5))   # время Казахстана (UTC+5)
CRUISE_KMH = 800                        # средняя скорость самолёта
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model", "forecat_model.joblib")
INFO_PATH = os.path.join(HERE, "model", "model_info.json")

AIRPORTS = {
    "Алматы (ALA)": (43.352, 77.040),
    "Астана (NQZ)": (51.022, 71.467),
    "Шымкент (CIT)": (42.364, 69.479),
    "Актау (SCO)": (43.860, 51.092),
    "Атырау (GUW)": (47.122, 51.821),
    "Актобе (AKX)": (50.246, 57.207),
    "Караганда (KGF)": (49.671, 73.334),
    "Усть-Каменогорск (UKK)": (50.037, 82.494),
    "Павлодар (PWQ)": (52.195, 77.074),
    "Костанай (KSN)": (53.207, 63.550),
    "Кызылорда (KZO)": (44.706, 65.593),
}
WEEKDAYS = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]


# ---------------------------------------------------------------------------
# Данные и модель
# ---------------------------------------------------------------------------

def model_stamp():
    """Время изменения файла модели: новый файл — модель загрузится заново."""
    return os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0


@st.cache_resource
def load_model(stamp):
    if not stamp:
        return None, None, None
    try:
        import joblib
        import sklearn
        bundle = joblib.load(MODEL_PATH)
        info = {}
        if os.path.exists(INFO_PATH):
            with open(INFO_PATH, encoding="utf-8") as f:
                info = json.load(f)
        warn = None
        if info.get("sklearn_version") and info["sklearn_version"] != sklearn.__version__:
            warn = (f"Модель обучена на scikit-learn {info['sklearn_version']}, а здесь стоит "
                    f"{sklearn.__version__}. Укажите нужную версию в requirements.txt.")
        return bundle, info, warn
    except Exception as e:  # повреждённый или несовместимый файл модели
        return None, None, f"Не удалось загрузить модель ИИ: {e}"


@st.cache_data(ttl=3600, show_spinner=False)
def route_weather(dep, arr):
    lat1, lon1 = AIRPORTS[dep]
    lat2, lon2 = AIRPORTS[arr]
    pts, total = tf.route_points(lat1, lon1, lat2, lon2)
    locations = []
    for lat, lon, _ in pts:
        locations += tf.stencil(lat, lon)
    frames = tf.fetch_openmeteo(locations, forecast_days=3)
    return pts, total, frames


def route_features(pts, frames, fl_ft):
    """Признаки для каждой точки маршрута на всех часах прогноза."""
    layer = tf.choose_layer(tf.fl_to_pressure(fl_ft))
    out = []
    for i, (lat, lon, km) in enumerate(pts):
        f = tf.compute_features(frames[5 * i: 5 * i + 5], lat, layer)
        f = f.assign(point=i, lat=lat, lon=lon, km=km)
        out.append(f)
    return pd.concat(out).rename_axis("time").reset_index()


def add_risk(feats, bundle, use_ai):
    feats = feats.copy()
    ok = feats[tf.FEATURES].notna().all(axis=1)
    feats["prob"] = np.nan
    if use_ai:
        model, cols = bundle["model"], bundle["features"]
        if ok.any():
            feats.loc[ok, "prob"] = model.predict_proba(feats.loc[ok, cols])[:, 1]
        feats["risk"] = tf.probability_to_risk(feats["prob"].fillna(0))
    else:
        feats["risk"] = tf.rule_risk(feats)
    feats.loc[~ok, "risk"] = -1
    return feats


def fmt_local(ts_utc):
    t = pd.Timestamp(ts_utc).tz_localize("UTC").tz_convert(KZ_TZ)
    return f"{WEEKDAYS[t.weekday()]} {t:%d.%m %H:%M}"


def hex_to_rgb(h, alpha=230):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [alpha]


# ---------------------------------------------------------------------------
# Интерфейс
# ---------------------------------------------------------------------------

st.title("✈️ ForeCAT-KZ")
st.markdown("##### Будет ли трясти в полёте? Прогноз турбулентности ясного неба над Казахстаном")

bundle, info, model_warn = load_model(model_stamp())

with st.sidebar:
    st.header("Рейс")
    names = list(AIRPORTS)
    dep = st.selectbox("Откуда", names, index=0)
    arr = st.selectbox("Куда", names, index=1)
    fl_ft = st.select_slider(
        "Высота полёта",
        options=list(range(30000, 41000, 1000)),
        value=35000,
        format_func=lambda x: f"{x:,} футов ≈ {x * 0.3048 / 1000:.1f} км".replace(",", " "),
    )
    modes = ["ИИ (обученная модель)", "Физическое правило (TI1 + Ri)"]
    mode = st.radio("Чем считать", modes, index=0 if bundle else 1, disabled=bundle is None)
    use_ai = bundle is not None and mode == modes[0]
    if bundle is None and model_warn:
        st.caption(f"⚠️ {model_warn}")
    elif bundle is None:
        st.caption("Модель ИИ не найдена в папке model/. Пока прогноз считается по физическому правилу.")

if dep == arr:
    st.info("Выберите разные аэропорты вылета и прилёта.")
    st.stop()
if model_warn and bundle is not None:
    st.warning(model_warn)

try:
    with st.spinner("Загружаю прогноз ветра на высотах полёта…"):
        pts, total_km, frames = route_weather(dep, arr)
except Exception as e:
    st.error(f"Не получилось загрузить прогноз Open-Meteo: {e}")
    st.stop()

feats = add_risk(route_features(pts, frames, fl_ft), bundle, use_ai)
flight_h = total_km / CRUISE_KMH
feats["offset_h"] = feats["km"] / CRUISE_KMH

# Возможные времена вылета: от ближайшего часа до конца прогноза
now = pd.Timestamp.now(tz="UTC").tz_localize(None).ceil("h")
hours = sorted(feats["time"].unique())
last_dep = pd.Timestamp(hours[-1]) - pd.Timedelta(hours=np.ceil(flight_h))
dep_options = [pd.Timestamp(h) for h in hours if now <= pd.Timestamp(h) <= last_dep]
if not dep_options:
    st.error("В прогнозе нет подходящих часов вылета. Обновите страницу позже.")
    st.stop()

with st.sidebar:
    labels = {fmt_local(d): d for d in dep_options}
    choice = st.select_slider("Вылет (время Казахстана)", options=list(labels),
                              value=list(labels)[min(3, len(labels) - 1)])
    dep_time = labels[choice]


def flight_snapshot(departure):
    """Для каждой точки маршрута берём час, когда самолёт будет над ней."""
    rows = []
    for i, g in feats.groupby("point"):
        when = departure + pd.Timedelta(hours=float(g["offset_h"].iloc[0]))
        j = (g["time"] - when).abs().idxmin()
        rows.append(feats.loc[j].to_dict() | {"over_time": when})
    return pd.DataFrame(rows)


snap = flight_snapshot(dep_time)
worst = int(snap["risk"].max())

# --- Итог по рейсу ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Расстояние", f"{total_km:,.0f} км".replace(",", " "))
c2.metric("В полёте", f"{int(flight_h)} ч {int(flight_h % 1 * 60)} мин")
c3.metric("Худший участок", tf.RISK_NAMES.get(worst, "нет данных"))
c4.metric("Чем считали", "ИИ" if use_ai else "Правило",
          help=(info or {}).get("model_name", "") if use_ai else "Индекс Эллрода TI1 и число Ричардсона")

color = tf.RISK_COLORS.get(worst, "#888888")
st.markdown(
    f"<div style='padding:14px 18px;border-radius:12px;background:{color}22;border-left:6px solid {color};"
    f"font-size:1.05rem'>"
    f"<b>{dep} → {arr}</b>, вылет {fmt_local(dep_time)}: "
    f"<b>{tf.RISK_NAMES.get(worst, 'нет данных')}</b> на высоте {fl_ft:,} футов</div>".replace(",", " "),
    unsafe_allow_html=True,
)

# --- Карта ---
snap["color"] = snap["risk"].map(lambda r: hex_to_rgb(tf.RISK_COLORS.get(int(r), "#888888")))
snap["risk_text"] = snap["risk"].map(lambda r: tf.RISK_NAMES.get(int(r), "нет данных"))
snap["when"] = snap["over_time"].map(fmt_local)
snap["prob_text"] = snap["prob"].map(lambda p: "" if pd.isna(p) else f" · вероятность {p:.0%}")
snap["vws_text"] = snap["vws"].round(1).astype(str)
snap["ri_text"] = snap["ri"].round(2).astype(str)
snap["ti1_text"] = snap["ti1"].round(1).astype(str)
(lat1, lon1), (lat2, lon2) = AIRPORTS[dep], AIRPORTS[arr]
path = [[lon1, lat1]] + [[lo, la] for la, lo in zip(snap["lat"], snap["lon"])] + [[lon2, lat2]]

deck = pdk.Deck(
    map_provider="carto",
    map_style=pdk.map_styles.CARTO_LIGHT,
    initial_view_state=pdk.ViewState(latitude=(lat1 + lat2) / 2, longitude=(lon1 + lon2) / 2,
                                     zoom=4.3 if total_km > 1200 else 5, pitch=0),
    layers=[
        pdk.Layer("PathLayer", data=[{"path": path}], get_path="path", get_color=[90, 110, 140, 160],
                  width_min_pixels=2),
        pdk.Layer("ScatterplotLayer", data=snap, get_position=["lon", "lat"], get_fill_color="color",
                  get_radius=22000, radius_min_pixels=7, pickable=True, stroked=True,
                  get_line_color=[255, 255, 255], line_width_min_pixels=1),
    ],
    tooltip={"html": "<b>{risk_text}</b>{prob_text}<br/>{when}<br/>сдвиг ветра {vws_text} уз/1000 фт · "
                     "Ri {ri_text} · TI1 {ti1_text}"},
)
st.pydeck_chart(deck)

# --- Когда лучше лететь ---
st.subheader("Когда лучше лететь")
st.caption("Для каждого часа вылета — самый неспокойный участок маршрута.")
by_dep = []
for d in dep_options:
    s = flight_snapshot(d)
    val = s["prob"].max() if use_ai else s["risk"].max()
    by_dep.append({"вылет": fmt_local(d), "t": d, "значение": val, "risk": int(s["risk"].max())})
by_dep = pd.DataFrame(by_dep)
by_dep["оценка"] = by_dep["risk"].map(lambda r: tf.RISK_NAMES.get(r, "нет данных"))
by_dep["t_local"] = by_dep["t"] + pd.Timedelta(hours=5)
x = alt.X("t_local:T", title="Время вылета (Казахстан)", axis=alt.Axis(format="%d.%m %H:%M", labelAngle=0))
risk_enc = alt.Color("оценка:N", title=None, legend=alt.Legend(orient="top", labelLimit=400),
                  scale=alt.Scale(domain=[tf.RISK_NAMES[k] for k in (0, 1, 2)],
                                  range=[tf.RISK_COLORS[k] for k in (0, 1, 2)]))
if use_ai:
    chart = alt.Chart(by_dep).mark_bar().encode(
        x=x, color=risk_enc,
        y=alt.Y("значение:Q", title="Вероятность болтанки", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%")),
        tooltip=["вылет", "оценка", alt.Tooltip("значение:Q", title="вероятность", format=".0%")],
    ).properties(height=320)
else:
    # Высота столбика = уровень риска: 1 — спокойно, 2 — возможна, 3 — сильная
    chart = alt.Chart(by_dep.assign(lvl=by_dep["risk"].clip(lower=0) + 1)).mark_bar().encode(
        x=x, color=risk_enc, tooltip=["вылет", "оценка"],
        y=alt.Y("lvl:Q", title="Уровень риска", scale=alt.Scale(domain=[0, 3]),
                axis=alt.Axis(values=[1, 2, 3], labels=False, ticks=False)),
    ).properties(height=320)
st.altair_chart(chart, width="stretch")

# --- Таблица по точкам ---
with st.expander("Подробно по точкам маршрута"):
    table = pd.DataFrame({
        "Км от вылета": snap["km"].round(0).astype(int),
        "Над точкой": snap["when"],
        "Оценка": snap["risk_text"],
        "Вероятность (ИИ)": snap["prob"].round(2) if use_ai else np.nan,
        "Ветер, м/с": snap["ws_mid"].round(0),
        "Сдвиг, уз/1000 фт": snap["vws"].round(1),
        "Ri": snap["ri"].round(2),
        "TI1": snap["ti1"].round(1),
    })
    if not use_ai:
        table = table.drop(columns=["Вероятность (ИИ)"])
    st.dataframe(table, hide_index=True, width="stretch")

with st.expander("Как это работает"):
    st.markdown(f"""
1. Берём свежий прогноз модели **GFS** (через Open-Meteo): ветер, температуру и высоту уровней
   300, 250 и 200 гПа (9–12 км) в точках вдоль маршрута и в соседних точках.
2. Считаем физику турбулентности ясного неба:
   - **сдвиг ветра** — насколько меняется ветер с высотой;
   - **число Ричардсона** Ri = N²/сдвиг² — побеждает устойчивость воздуха или сдвиг;
   - **индекс Эллрода** TI1 = сдвиг × деформация потока.
3. **Физическое правило:** TI1 ≥ 4 или Ri < 1 — «возможна болтанка»; TI1 ≥ 8 или Ri < 0,5 — «сильная болтанка вероятна».
4. **ИИ:** модель обучена на {(info or {}).get('n_samples', 'тысяче')} отчётах пилотов (PIREP)
   и сама находит сочетания признаков. Вероятность — это уровень риска: модель училась на выборке,
   где болтанки и спокойных полётов поровну, поэтому в жизни болтанка встречается реже.

**Ограничения:**
- модель погоды видит детали размером около 25 км, а вихри турбулентности — сотни метров;
- грозовую турбулентность этот прогноз не учитывает;
- это школьный исследовательский проект, а не инструмент для пилотов.
""")

st.caption("Данные: Open-Meteo (модель NOAA GFS), отчёты пилотов — Iowa Environmental Mesonet. "
           "Идея — проект ForeCAT (Regeneron ISEF 2026). Проект для конкурса «Зерде».")
