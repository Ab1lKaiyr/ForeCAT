"""
ForeCAT-KZ — страница «Самолёт-лаборатория».

  • «Сейчас»: живые данные с ESP32 в модели самолёта (обновляются каждую секунду):
      напрямую по Wi-Fi или через интернет (самолёт кладёт их в облачную базу Firebase);
  • «Журнал и опыты»: разбор журнала полёта или опытов A–E на стенде
    и проверка, насколько дешёвые датчики удара совпадают с акселерометром.
"""

import io
import time
from collections import deque

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

LEVEL_NAMES = ["спокойно", "слабая", "умеренная", "сильная"]
LEVEL_COLORS = ["#2e9e5b", "#c99700", "#e8590c", "#d64545"]
MODES = {
    "A": "A · без ветра (фон)",
    "B": "B · ровный поток",
    "C": "C · препятствие",
    "D": "D · сдвиг ветра",
    "E": "E · тёплый поток",
}
MODE_HELP = {
    "A": "вентилятор выключен — узнаём шум датчика",
    "B": "один вентилятор дует прямо на модель",
    "C": "между вентилятором и моделью решётка или книга",
    "D": "два вентилятора разной силы, модель на границе потоков — аналог CAT",
    "E": "фен: тёплый поднимающийся воздух — конвекция",
}
MODE_COLORS = ["#8a96a3", "#2563eb", "#7c3aed", "#d64545", "#e8590c"]
LOG_COLUMNS = ["session", "t_s", "mode", "level", "source", "peak_g", "sigma_g",
               "shock_per_s", "knock_per_s", "temp_c", "airspeed_ms"]
DEFAULT_ADDR = "http://forecat.local"


def normalize(addr):
    addr = (addr or "").strip().rstrip("/")
    if addr and not addr.startswith(("http://", "https://")):
        addr = "http://" + addr
    return addr


def call(addr, path, **params):
    r = requests.get(f"{addr}{path}", params=params, timeout=(2, 4))
    r.raise_for_status()
    return r


def fmt(v, dec, unit=""):
    return "—" if v is None else f"{float(v):.{dec}f}{unit}"


def ago(sec):
    sec = max(0, int(sec))
    if sec < 90:
        return f"{sec} с"
    if sec < 90 * 60:
        return f"{sec // 60} мин"
    if sec < 36 * 3600:
        return f"{sec // 3600} ч"
    return f"{sec // 86400} дн"


def secret(name):
    """Значение из секретов сайта (на Streamlit Cloud: Settings → Secrets)."""
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


CLOUD_URL = secret("FORECAT_CLOUD_URL").rstrip("/")
CLOUD_KEY = secret("FORECAT_CLOUD_KEY")
WAYS = ["Через интернет", "Напрямую по Wi-Fi"]


ss = st.session_state
st.title("🛩️ Самолёт-лаборатория")
st.caption("Модель самолёта с ESP32 и датчиками — наш маленький «CATalog»: "
           "прибор, который сам измеряет турбулентность и предупреждает о ней.")

with st.sidebar:
    st.header("Связь с самолётом")
    way = st.radio("Откуда брать данные", WAYS, index=0 if CLOUD_URL and CLOUD_KEY else 1,
                   help="Через интернет — самолёт сам отправляет показания в облачную базу, "
                        "их видно с любого телефона. Напрямую — ноутбук в одной сети с самолётом: "
                        "так работают ещё кнопки опытов и загрузка журнала.")
    cloud = way == WAYS[0]
    addr = ss.get("plane_addr", DEFAULT_ADDR)
    if cloud:
        if CLOUD_URL and CLOUD_KEY:
            cloud_url, cloud_key = CLOUD_URL, CLOUD_KEY
            st.caption("Адрес базы и секретное слово взяты из секретов сайта.")
        else:
            cloud_url = st.text_input("Адрес базы Firebase", value=ss.get("cloud_url", "")).strip().rstrip("/")
            cloud_key = st.text_input("Секретное слово", value=ss.get("cloud_key", ""), type="password").strip()
            ss["cloud_url"], ss["cloud_key"] = cloud_url, cloud_key
            st.caption("Те же значения, что CLOUD_URL и CLOUD_KEY в config.h самолёта.")
        src = ("cloud", cloud_url, cloud_key)
    else:
        addr = normalize(st.text_input("Адрес самолёта", value=addr))
        ss["plane_addr"] = addr
        st.caption("Адрес написан на нижней строке экрана самолёта.\n\n"
                   "• Сеть телефона: http://forecat.local или IP с экрана\n\n"
                   "• Сеть самолёта ForeCAT-Plane: http://192.168.4.1")
        src = ("direct", addr)
    live_on = st.toggle("Следить за самолётом", value=True)

tab_live, tab_log = st.tabs(["📡 Сейчас", "📒 Журнал и опыты"])


# ---------------------------------------------------------------------------
# Сейчас: живые данные
# ---------------------------------------------------------------------------

def send_mode(m):
    try:
        call(ss["plane_addr"], "/mode", m=m)
        st.toast(f"Метка опыта: {MODES.get(m, 'без метки')}")
    except Exception:
        st.toast("Самолёт не ответил")


def send_cmd(path, text):
    try:
        call(ss["plane_addr"], path)
        st.toast(text)
    except Exception:
        st.toast("Самолёт не ответил")


class NoData(Exception):
    pass


def fetch(src):
    """Показания самолёта: напрямую с ESP32 или из облачной базы."""
    if src[0] == "direct":
        return call(src[1], "/data", since=ss["since"]).json()
    _, url, key = src
    r = requests.get(f"{normalize(url)}/{key}/live.json", timeout=(3, 6))
    if r.status_code in (401, 403):
        raise PermissionError
    r.raise_for_status()
    d = r.json()
    if not isinstance(d, dict) or "level" not in d:
        raise NoData
    return d


@st.fragment(run_every=1.0)
def live_panel(src, live_on):
    if ss.get("live_src") != src:
        ss["live_src"] = src
        ss["since"] = 0
        ss["wave"] = deque(maxlen=500)
    if not live_on:
        st.info("Слежение на паузе.")
        return
    if src[0] == "cloud" and not (src[1] and src[2]):
        st.info("Впишите слева адрес базы Firebase и секретное слово — те же, что в config.h самолёта.")
        return
    try:
        d = fetch(src)
    except PermissionError:
        st.warning("База не пускает: проверьте секретное слово и правила базы "
                   "(README_samolet.md, раздел «Сайт в интернете»).")
        return
    except NoData:
        st.info("В базе пока нет показаний. Включите самолёт: в config.h должны быть CLOUD_URL и CLOUD_KEY, "
                "а телефон должен раздавать интернет.")
        return
    except Exception:
        if src[0] == "cloud":
            st.warning("Нет связи с облачной базой. Проверьте адрес базы и интернет.")
        else:
            st.warning(
                f"Самолёт не на связи по адресу {src[1]}.\n\n"
                "Проверьте: плата включена; ноутбук в той же сети, что и самолёт "
                "(точка доступа телефона или сеть ForeCAT-Plane); адрес совпадает с экраном самолёта. "
                "Если адрес forecat.local не открывается, впишите IP-адрес с экрана."
            )
        return

    age = None
    # Firebase не хранит пустые значения (null): датчика нет — поля просто нет, поэтому везде d.get()
    if src[0] == "cloud" and isinstance(d.get("updated"), (int, float)):
        age = time.time() - d["updated"] / 1000
        if age > 10:
            st.warning(f"Самолёт сейчас не в сети. Ниже последние показания — они пришли {ago(age)} назад.")

    # новые точки волны ускорения (номер wave_seq растёт на 25 в секунду)
    seq = int(d.get("wave_seq", 0))
    pts = [float(x) for x in d.get("wave", [])]
    if seq < ss["since"]:  # плата перезагрузилась
        ss["wave"].clear()
        ss["since"] = 0
    new = len(pts) if ss["since"] == 0 else max(0, min(len(pts), seq - ss["since"]))
    if new:
        ss["wave"].extend(pts[-new:])
    ss["since"] = seq

    lvl = int(d["level"])
    color = LEVEL_COLORS[lvl]
    source = {"fc": "по акселерометру полётного контроллера F405-WING",
              "accel": "по акселерометру MPU6050",
              "shock": "акселерометра нет — оценка по датчикам удара KY-002 и KY-031",
              "none": "датчиков тряски нет"}[d["source"]]
    mode = d.get("mode", "-")
    st.markdown(
        f"<div style='padding:14px 18px;border-radius:12px;background:{color}22;border-left:8px solid {color}'>"
        f"<div style='font-size:.9rem;opacity:.75'>Турбулентность сейчас · {source}"
        f"{' · опыт ' + MODES[mode] if mode in MODES else ''}</div>"
        f"<div style='font-size:2rem;font-weight:800;color:{color}'>{d['level_name'].upper()}</div></div>",
        unsafe_allow_html=True,
    )
    c = st.columns(6)
    c[0].metric("Пик за 1 с", fmt(d.get("peak_g"), 2, " g"))
    c[1].metric("Сила тряски (СКО)", fmt(d.get("sigma_g"), 2, " g"))
    c[2].metric("Максимум за сеанс", fmt(d.get("peak_max_g"), 2, " g"))
    c[3].metric("Поток воздуха", fmt(d.get("airspeed_ms"), 1, " м/с"))
    c[4].metric("Температура", fmt(d.get("temp_c"), 1, " °C"))
    c[5].metric("Удар / стук в с", f"{fmt(d.get('shock_per_s'), 0)} / {fmt(d.get('knock_per_s'), 0)}")

    wave = list(ss["wave"])
    if d["source"] in ("accel", "fc") and wave:
        hz = d.get("wave_hz", 25)
        w = pd.DataFrame({"t": (np.arange(len(wave)) - len(wave) + 1) / hz, "g": wave})
        thr = [float(x) for x in d["thresholds_g"]]
        rules = pd.DataFrame([{"y": s * t, "порог": n} for t, n in zip(thr, LEVEL_NAMES[1:]) for s in (1, -1)])
        lim = max(thr[2] * 1.25, float(np.abs(w["g"]).max()) * 1.05)
        line = alt.Chart(w).mark_line(strokeWidth=1.5).encode(
            x=alt.X("t:Q", title="секунды назад", scale=alt.Scale(domain=[-20, 0])),
            y=alt.Y("g:Q", title="вертикальное ускорение порыва, g", scale=alt.Scale(domain=[-lim, lim])),
        )
        rule = alt.Chart(rules).mark_rule(strokeDash=[5, 4]).encode(
            y="y:Q",
            color=alt.Color("порог:N", scale=alt.Scale(domain=LEVEL_NAMES[1:], range=LEVEL_COLORS[1:]),
                            legend=alt.Legend(orient="top", title="пороги")),
        )
        st.altair_chart((rule + line).properties(height=260), width="stretch")
    fc = d.get("fc") or {}
    if fc.get("connected"):
        f = st.columns(4)
        f[0].metric("Высота (барометр)", fmt(fc.get("alt_m"), 0, " м"))
        f[1].metric("Вертикальная скорость", fmt(fc.get("vario_ms"), 1, " м/с"))
        f[2].metric("Крен / тангаж", f"{fmt(fc.get('roll_deg'), 0, '°')} / {fmt(fc.get('pitch_deg'), 0, '°')}")
        f[3].metric("Батарея", fmt(fc.get("vbat_v"), 1, " В") if fc.get("vbat_v") is not None else "от USB")
    elif (d.get("sensors") or {}).get("fc"):
        st.caption("Полётный контроллер не отвечает: проверьте провода TX6→RX2, RX6←TX2, GND "
                   "и что на UART6 в INAV включён MSP.")
    elif d["source"] not in ("accel", "fc"):
        st.caption("График ускорения появится, когда будет подключён акселерометр MPU6050.")

    tl = [float(x) for x in d["time_in_level_s"]]
    if sum(tl) > 0:
        st.caption("Время на каждом уровне за сеанс: " + " · ".join(
            f"{n} {t:.0f} с" for n, t in zip(LEVEL_NAMES, tl)))
    net = d.get("wifi") or {}
    cl = d.get("cloud") or {}
    st.caption(
        (f"Через интернет · обновлено {ago(age)} назад · " if age is not None else "")
        + f"Сеанс №{d['session']} · журнал: {'идёт запись' if d['log_on'] else 'запись остановлена'}, "
        f"{d['log_kb']} КБ · Telegram: "
        + (f"включён, подписчиков {d['telegram_subscribers']}" if d["telegram"] else "выключен")
        + (f" · в сети телефона: {net['ip']}"
           if src[0] == "direct" and net.get("phone_connected") and net.get("ip") else "")
        + ((" · в интернет: " + ("отправка идёт" if cl.get("ok") else "нет связи"))
           if src[0] == "direct" and cl.get("on") else "")
    )


with tab_live:
    live_panel(src, live_on)
    if cloud:
        st.caption("Кнопки опытов на стенде и загрузка журнала работают при связи «Напрямую по Wi-Fi»: "
                   "ноутбук в одной сети с самолётом.")
    else:
        st.markdown("**Метка опыта на стенде** — записывается в журнал, чтобы потом сравнить опыты:")
        cols = st.columns(len(MODES) + 1)
        cols[0].button("Без метки", on_click=send_mode, args=("-",), width="stretch")
        for col, (m, name) in zip(cols[1:], MODES.items()):
            col.button(name, on_click=send_mode, args=(m,), help=MODE_HELP[m], width="stretch")
        b1, b2, b3, _ = st.columns([1, 1, 1, 2])
        b1.button("Сбросить максимум", on_click=send_cmd, args=("/reset", "Максимум сброшен"), width="stretch")
        b2.button("Начать запись", on_click=send_cmd, args=("/log/start", "Запись идёт"), width="stretch")
        b3.button("Остановить запись", on_click=send_cmd, args=("/log/stop", "Запись остановлена"),
                  width="stretch")


# ---------------------------------------------------------------------------
# Журнал и опыты
# ---------------------------------------------------------------------------

def read_log(raw):
    df = pd.read_csv(io.BytesIO(raw))
    missing = [c for c in LOG_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("в файле нет столбцов: " + ", ".join(missing))
    for c in df.columns:
        if c not in ("mode", "source"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["mode"] = df["mode"].fillna("-").astype(str).str.strip().str.upper()
    return df.dropna(subset=["session", "t_s", "level"])


def correlation_words(r):
    a = abs(r)
    return "сильная" if a >= 0.7 else "заметная" if a >= 0.4 else "слабая"


def analyze(df):
    sessions = sorted(int(s) for s in df["session"].unique())
    chosen = st.multiselect("Сеансы (каждое включение платы — новый сеанс)", sessions,
                            default=sessions[-1:])
    d = df[df["session"].isin(chosen)].copy()
    if d.empty:
        st.info("Выберите хотя бы один сеанс.")
        return

    accel = d["peak_g"].notna().any()
    if accel:
        d["strength"] = d["sigma_g"]
        d["intensity"] = d["peak_g"]
        s_name, i_name = "Сила тряски (СКО), g", "Пик за 1 с, g"
    else:
        d["strength"] = d["shock_per_s"].fillna(0) + d["knock_per_s"].fillna(0)
        d["intensity"] = d["strength"]
        s_name = i_name = "Срабатываний датчиков удара в секунду"
        st.info("В этих сеансах не было акселерометра — анализ по датчикам удара KY-002 и KY-031.")

    # --- общая картина ---
    dur = d.groupby("session")["t_s"].agg(lambda s: s.max() - s.min()).sum()
    k = st.columns(4)
    k[0].metric("Записано", f"{dur / 60:.1f} мин")
    k[1].metric("Максимум", f"{d['intensity'].max():.2f}" + (" g" if accel else " /с"))
    k[2].metric("Время «умеренная» и сильнее", f"{(d['level'] >= 2).mean() * 100:.0f} %")
    if d["temp_c"].notna().any():
        k[3].metric("Температура", f"{d['temp_c'].min():.1f}…{d['temp_c'].max():.1f} °C")

    # --- график сеанса ---
    one = st.selectbox("Показать на графике сеанс", chosen, index=len(chosen) - 1)
    s1 = d[d["session"] == one].sort_values("t_s").copy()
    s1["уровень"] = s1["level"].astype(int).map(dict(enumerate(LEVEL_NAMES)))
    seg_id = (s1["mode"] != s1["mode"].shift()).cumsum()
    segs = s1.groupby(seg_id).agg(start=("t_s", "min"), end=("t_s", "max"), mode=("mode", "first"))
    segs = segs[segs["mode"].isin(MODES)].assign(опыт=lambda x: x["mode"].map(MODES))
    base = alt.Chart(s1).encode(x=alt.X("t_s:Q", title="время с включения, с"))
    layers = []
    if len(segs):
        layers.append(alt.Chart(segs).mark_rect(opacity=0.12).encode(
            x="start:Q", x2="end:Q",
            color=alt.Color("опыт:N", scale=alt.Scale(domain=list(MODES.values()), range=MODE_COLORS),
                            legend=alt.Legend(orient="bottom", title=None))))
    layers.append(base.mark_line(strokeWidth=1, color="#8a96a3").encode(y=alt.Y("intensity:Q", title=i_name)))
    layers.append(base.mark_circle(size=14, opacity=0.9).encode(
        y="intensity:Q",
        color=alt.Color("уровень:N", scale=alt.Scale(domain=LEVEL_NAMES, range=LEVEL_COLORS),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("t_s:Q", title="время, с"), "уровень",
                 alt.Tooltip("intensity:Q", title=i_name, format=".3f"), "mode"]))
    st.altair_chart(alt.layer(*layers).resolve_scale(color="independent").properties(height=300),
                    width="stretch")
    if "vario_ms" in s1 and s1["vario_ms"].notna().any():
        st.altair_chart(alt.Chart(s1).mark_line(strokeWidth=1, color="#7c3aed").encode(
            x=alt.X("t_s:Q", title="время с включения, с"),
            y=alt.Y("vario_ms:Q", title="вертикальная скорость, м/с"),
        ).properties(height=160), width="stretch")
        st.caption("Вертикальная скорость по барометру полётного контроллера: в болтанку самолёт "
                   "то подбрасывает, то роняет.")

    # --- сравнение опытов ---
    st.subheader("Сравнение опытов на стенде")
    d = d.sort_values(["session", "t_s"])
    d["seg"] = ((d["mode"] != d["mode"].shift()) | (d["session"] != d["session"].shift())).cumsum()
    d["t_in"] = d["t_s"] - d.groupby("seg")["t_s"].transform("min")
    x = d[d["mode"].isin(MODES) & (d["t_in"] >= 1.5)]   # первые 1,5 с после смены опыта не берём
    if x.empty:
        st.info("В журнале нет меток опытов A–E. Нажимайте кнопку опыта на сайте самолёта "
                "перед каждой серией — тогда здесь появится сравнение.")
        return
    per_seg = x.groupby(["mode", "seg"]).agg(
        strength=("strength", "mean"), intensity=("intensity", "mean"),
        dur=("t_in", "max"), mod_share=("level", lambda l: (l >= 2).mean()),
        temp=("temp_c", "mean"), air=("airspeed_ms", "mean"),
    ).reset_index()
    per_mode = per_seg.groupby("mode").agg(
        series=("seg", "size"), dur=("dur", "sum"), mean=("strength", "mean"),
        spread=("strength", "std"), peak=("intensity", "mean"), mod_share=("mod_share", "mean"),
        temp=("temp", "mean"), air=("air", "mean"),
    ).reset_index()
    per_mode["опыт"] = per_mode["mode"].map(MODES)
    per_mode["spread"] = per_mode["spread"].fillna(0)

    bars = alt.Chart(per_mode).mark_bar().encode(
        x=alt.X("опыт:N", sort=list(MODES.values()), title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("mean:Q", title=s_name),
        color=alt.Color("опыт:N", scale=alt.Scale(domain=list(MODES.values()), range=MODE_COLORS), legend=None),
        tooltip=["опыт", alt.Tooltip("mean:Q", title="среднее", format=".3f"),
                 alt.Tooltip("spread:Q", title="разброс между сериями", format=".3f"), "series"],
    )
    err = alt.Chart(per_mode.assign(lo=lambda t: t["mean"] - t["spread"], hi=lambda t: t["mean"] + t["spread"])
                    ).mark_errorbar(ticks=True).encode(
        x=alt.X("опыт:N", sort=list(MODES.values())), y=alt.Y("lo:Q", title=s_name), y2="hi:Q")
    st.altair_chart((bars + err).properties(height=300), width="stretch")
    st.caption("Столбик — среднее по сериям, чёрточки — разброс между сериями (стандартное отклонение).")

    table = pd.DataFrame({
        "Опыт": per_mode["опыт"],
        "Серий": per_mode["series"],
        "Секунд": per_mode["dur"].round(0).astype(int),
        s_name: per_mode["mean"].round(3),
        "± разброс": per_mode["spread"].round(3),
        i_name + " (средний)": per_mode["peak"].round(3),
        "Доля «умеренная+», %": (per_mode["mod_share"] * 100).round(0),
        "Поток, м/с": per_mode["air"].round(1),
        "Температура, °C": per_mode["temp"].round(1),
    }).dropna(axis=1, how="all")
    if not accel:
        table = table.drop(columns=[i_name + " (средний)"])
    st.dataframe(table, hide_index=True, width="stretch")

    # --- выводы словами ---
    m = per_mode.set_index("mode")["mean"]
    notes = []
    unit = " g" if accel else " срабатываний/с"
    if "A" in m:
        notes.append(f"Фон без ветра (A): {m['A']:.3f}{unit}. Это шум прибора: всё, что не больше, "
                     "— не турбулентность.")
    base_mode = "B" if "B" in m else ("A" if "A" in m else None)
    for mm, what in (("C", "С препятствием"), ("D", "Со сдвигом ветра"), ("E", "С тёплым потоком")):
        if base_mode and mm in m and m[base_mode] > 0:
            notes.append(f"{what} ({mm}) тряска в {m[mm] / m[base_mode]:.1f} раза сильнее, "
                         f"чем в опыте {base_mode}.")
    t = per_mode.set_index("mode")["temp"]
    if "E" in t and "B" in t and pd.notna(t["E"]) and pd.notna(t["B"]):
        notes.append(f"Тёплый поток (E) теплее ровного (B) на {t['E'] - t['B']:.1f} °C.")
    if (per_mode["series"] < 3).any():
        notes.append("Чтобы вывод был надёжным, повторите каждый опыт хотя бы 3 раза по 30 секунд.")
    for n in notes:
        st.markdown("• " + n)

    # --- дешёвые датчики против акселерометра ---
    if accel:
        cmp_ = d[d["source"].isin(["accel", "fc"])].copy()
        cmp_["удары"] = cmp_["shock_per_s"].fillna(0) + cmp_["knock_per_s"].fillna(0)
        has_shock = cmp_["shock_per_s"].notna().any() or cmp_["knock_per_s"].notna().any()
        if has_shock and cmp_["удары"].std() > 0 and cmp_["peak_g"].std() > 0:
            st.subheader("Дешёвые датчики удара против акселерометра")
            r = float(np.corrcoef(cmp_["удары"], cmp_["peak_g"])[0, 1])
            st.altair_chart(alt.Chart(cmp_).mark_circle(size=18, opacity=0.4).encode(
                x=alt.X("удары:Q", title="срабатываний KY-002 + KY-031 в секунду"),
                y=alt.Y("peak_g:Q", title="пик ускорения по MPU6050, g"),
            ).properties(height=260), width="stretch")
            st.markdown(f"Коэффициент корреляции r = **{r:.2f}** — связь {correlation_words(r)}. "
                        "Датчики удара только говорят «был толчок / не было», а акселерометр "
                        "измеряет силу толчка в g, поэтому для оценки турбулентности он точнее.")


with tab_log:
    log_src = st.radio("Откуда взять журнал", ["Из файла"] if cloud else ["С самолёта", "Из файла"],
                       horizontal=True)
    if log_src == "С самолёта":
        if st.button("Загрузить журнал с самолёта"):
            try:
                r = requests.get(f"{addr}/log.csv", timeout=(3, 120))
                if r.status_code == 404:
                    st.warning("Журнал на самолёте пуст.")
                else:
                    r.raise_for_status()
                    ss["log_bytes"] = r.content
            except Exception:
                st.warning(f"Не получилось скачать журнал с {addr}. Проверьте связь (вкладка «Сейчас»).")
    else:
        up = st.file_uploader("Файл журнала forecat_log.csv", type="csv")
        if up is not None:
            ss["log_bytes"] = up.getvalue()

    raw = ss.get("log_bytes")
    if raw:
        st.download_button("💾 Сохранить журнал на компьютер", raw, file_name="forecat_log.csv",
                           mime="text/csv")
        try:
            df = read_log(raw)
        except Exception as e:
            st.error(f"Не получилось прочитать журнал: {e}")
        else:
            if df.empty:
                st.info("Журнал пустой.")
            else:
                analyze(df)
    else:
        st.caption("Журнал записывается на самолёте сам, 5 строк в секунду. "
                   "Загрузите его сюда после полёта или серии опытов.")
