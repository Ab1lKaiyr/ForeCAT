"""
ForeCAT-KZ — живые данные самолёта-лаборатории.

Самолёт (ESP32) через точку доступа телефона раз в секунду кладёт измерения
в базу Firebase Realtime Database (раздел /<DB_PATH>/log). Этот модуль их читает
и показывает на странице «Самолёт-лаборатория».

Подключение в plane.py — две строки в том месте страницы, где нужен блок:
    from live import show_live
    show_live()
"""

from datetime import timedelta, timezone

import altair as alt
import pandas as pd
import requests
import streamlit as st

# Адрес базы — тот же, что DB_URL в прошивке.
FIREBASE_URL = "https://catar-ecffd-default-rtdb.europe-west1.firebasedatabase.app/"
# Тайное слово — раздел базы, открытый в правилах Firebase. Тот же, что DB_PATH в прошивке.
DB_PATH = "samolet-7f3k9q2x"

THR = (0.05, 0.15, 0.35)   # пороги из прошивки: слабая, умеренная, сильная, g
LEVELS = [("🟢", "Спокойно"), ("🟡", "Слабая"), ("🔴", "Умеренная"), ("🚨", "Сильная")]
OFFLINE_S = 10             # если новых данных нет дольше — самолёт не на связи
UTC5 = timezone(timedelta(hours=5), "UTC+5")   # Казахстан
COLUMNS = ["ts", "ses", "lvl", "pk", "sko", "shk", "tot", "t", "mpu", "rssi", "up", "clk"]


@st.cache_data(ttl=2, show_spinner=False)
def fetch_log(url: str, n: int) -> pd.DataFrame:
    """Последние n записей из базы.

    Кэш общий для всех зрителей: сколько бы человек ни открыло сайт по QR,
    база получает один запрос раз в 2 секунды.
    """
    r = requests.get(f"{url}/{DB_PATH}/log.json",
                     params={"orderBy": '"$key"', "limitToLast": n}, timeout=6)
    r.raise_for_status()
    data = r.json() or {}
    # Ключи записей Firebase растут со временем, поэтому сортировка по ключу = по времени.
    rows = [v for _, v in sorted(data.items()) if isinstance(v, dict)]
    df = pd.DataFrame(rows).reindex(columns=COLUMNS)   # пустые поля (null) Firebase не хранит
    for col in ("ts", "lvl", "pk", "sko", "shk", "tot", "t", "rssi", "up"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ts"])
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(UTC5)
    return df


def _num(x, fmt, unit="", empty="—"):
    return empty if pd.isna(x) else f"{x:{fmt}}{unit}"


def _ago(sec):
    sec = int(sec)
    if sec < 60:
        return f"{sec} с"
    if sec < 3600:
        return f"{sec // 60} мин"
    if sec < 86400:
        return f"{sec // 3600} ч"
    return f"{sec // 86400} дн"


def _shake_chart(df):
    """Пик и СКО по времени + пунктиром пороги «умеренная» и «сильная»."""
    long = df.melt(id_vars="up", value_vars=["pk", "sko"], var_name="ряд", value_name="g")
    long["ряд"] = long["ряд"].map({"pk": "Пик за секунду", "sko": "СКО"})
    biggest = long["g"].max()
    top = max(THR[2] * 1.15, 0 if pd.isna(biggest) else float(biggest) * 1.1)
    y = alt.Y("g:Q", title="Ускорение порыва, g", scale=alt.Scale(domain=[0, top]))

    lines = alt.Chart(long).mark_line().encode(
        x=alt.X("up:Q", title="Время с включения самолёта, с"),
        y=y,
        color=alt.Color("ряд:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=["Пик за секунду", "СКО"],
                                        range=["#2563eb", "#94a3b8"])),
        tooltip=[alt.Tooltip("up:Q", title="с"), "ряд:N", alt.Tooltip("g:Q", format=".3f")],
    )

    def threshold(value, color, text):
        base = alt.Chart(pd.DataFrame({"g": [value]}))
        rule = base.mark_rule(color=color, strokeDash=[6, 4]).encode(y=y)
        label = base.mark_text(color=color, align="left", dx=4, dy=-7, fontSize=11).encode(
            y=y, x=alt.value(0), text=alt.value(text))
        return rule + label

    return (lines
            + threshold(THR[1], "#d97706", "умеренная")
            + threshold(THR[2], "#dc2626", "сильная")).properties(height=290)


def show_live(points: int = 180):
    """Блок «Самолёт в прямом эфире». points — сколько последних секунд показывать."""
    st.subheader("📡 Самолёт в прямом эфире")
    url = FIREBASE_URL.strip().rstrip("/")
    if not url:
        st.info("Вставьте адрес базы Firebase в FIREBASE_URL в файле live.py.")
        return
    _live_panel(url, points)


@st.fragment(run_every=2)   # обновляется сам каждые 2 секунды, остальная страница не трогается
def _live_panel(url, points):
    try:
        df = fetch_log(url, points)
    except Exception as e:
        if getattr(getattr(e, "response", None), "status_code", None) == 401:
            st.error("База не пускает (401): DB_PATH в live.py не совпадает с правилами Firebase "
                     "(вкладка Rules).")
        else:
            st.warning(f"База данных сейчас не отвечает ({type(e).__name__}). Пробую снова…")
        return
    if df.empty:
        st.info("Данных пока нет. Включите самолёт и точку доступа на телефоне.")
        return

    last = df.iloc[-1]
    df = df[df["ses"] == last["ses"]]          # только текущее включение самолёта
    age = (pd.Timestamp.now(tz="UTC") - last["time"]).total_seconds()

    if age > OFFLINE_S:
        st.warning(f"Самолёт не на связи: последние данные пришли {_ago(age)} назад "
                   f"({last['time']:%d.%m %H:%M:%S}). Ниже — то, что он передал последним.")
    else:
        st.caption(f"🟢 На связи · обновление каждую секунду · {last['time']:%H:%M:%S} (UTC+5)")

    lvl = 0 if pd.isna(last["lvl"]) else int(min(max(last["lvl"], 0), 3))
    icon, name = LEVELS[lvl]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Тряска", f"{icon} {name}")
    c2.metric("Пик за секунду", _num(last["pk"], ".3f", " g"))
    c3.metric("СКО (средняя тряска)", _num(last["sko"], ".3f", " g"))
    c4.metric("Температура воздуха", _num(last["t"], ".1f", " °C"))

    if pd.notna(last["mpu"]) and not bool(last["mpu"]):
        st.caption("MPU6050 не найден — уровень тряски считается по толчкам KY-002.")
    st.altair_chart(_shake_chart(df))

    with st.expander("Температура и толчки KY-002"):
        a, b = st.columns(2)
        a.line_chart(df.set_index("up")[["t"]].rename(columns={"t": "°C"}), height=180,
                     x_label="с", y_label="°C")
        b.line_chart(df.set_index("up")[["shk"]].rename(columns={"shk": "толчков/с"}), height=180,
                     x_label="с", y_label="толчков/с")

    st.caption(f"Самолёт включён {_num(last['up'], '.0f', ' с')} · "
               f"толчков KY-002 всего: {_num(last['tot'], '.0f')} · "
               f"сигнал Wi-Fi: {_num(last['rssi'], '.0f', ' дБм')} · "
               f"часы самолёта: {last['clk'] if isinstance(last['clk'], str) and last['clk'] else '—'}")

    out = df.drop(columns=["ts"]).assign(time=df["time"].dt.strftime("%Y-%m-%d %H:%M:%S"))
    st.download_button("Скачать эти данные (CSV)", out.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"forecat_polet_{last['ses']}.csv", mime="text/csv")
