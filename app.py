"""
ForeCAT-KZ — сайт проекта.

Запуск на компьютере:   python -m streamlit run app.py
Страницы:
  • «Прогноз рейса»       — forecast.py: физика + ИИ по прогнозу погоды;
  • «Самолёт-лаборатория» — plane.py: живые данные с модели самолёта и разбор опытов.
Внизу боковой панели — QR-код сайта для постера.
"""

import io
from urllib.parse import urlsplit

import segno
import streamlit as st
from live import show_live
show_live()

st.set_page_config(page_title="ForeCAT-KZ", page_icon="✈️", layout="wide")

pages = [
    st.Page("forecast.py", title="Прогноз рейса", icon="🗺️", default=True),
    st.Page("plane.py", title="Самолёт-лаборатория", icon="🛩️"),
]
st.navigation(pages).run()


@st.cache_data
def qr_png(url):
    buf = io.BytesIO()
    segno.make(url, error="m").save(buf, kind="png", scale=12, border=3, dark="#1f2933")
    return buf.getvalue()


def site_root():
    """Адрес сайта без названия страницы, например https://forecat-kz.streamlit.app/"""
    try:
        u = urlsplit(st.context.url or "")
    except Exception:
        return ""
    return f"{u.scheme}://{u.netloc}/" if u.scheme and u.netloc else ""


with st.sidebar:
    st.divider()
    with st.expander("📱 QR-код сайта для постера"):
        url = st.text_input("Адрес сайта", value=site_root(),
                            help="Опубликованный адрес вида https://….streamlit.app")
        if url.strip():
            png = qr_png(url.strip())
            st.image(png, width=200)
            st.download_button("Скачать QR-код (PNG)", png, file_name="forecat_qr.png", mime="image/png",
                               width="stretch")
            if any(h in url for h in ("localhost", "127.0.0.1", "192.168.")):
                st.caption("Это адрес на вашем компьютере, с телефона жюри он не откроется. "
                           "Сделайте QR на опубликованном сайте (…streamlit.app).")
