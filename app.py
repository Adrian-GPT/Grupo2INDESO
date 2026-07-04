# app.py — Ernesto Investing AI (Bono Streamlit, Semana 13)
# ------------------------------------------------------------------
# App 100% Python que lee datos YA CALCULADOS de MongoDB Atlas
# (no recalcula nada: mismas colecciones que usa la API FastAPI del
# Notebook 3) y los muestra con st.plotly_chart, st.metric y
# st.selectbox, como pide el Anexo E / Ejercicio 4 de la guia.
#
# Colecciones leidas (pobladas por Notebook 1 y Notebook 2):
#   - precios_ohlcv     -> ticker, fecha, open, high, low, close, volume,
#                          sma_20, sma_50, ema_12, ema_26, rsi_14
#   - predicciones       -> ticker, modelo, senal, confianza, fecha_prediccion
#   - metricas_modelos   -> ticker, modelo, accuracy, precision, recall, f1,
#                          mejor_kernel, mejor_C, mejor_gamma, matriz_confusion
# ------------------------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymongo import MongoClient
from pymongo.server_api import ServerApi

# ------------------------------------------------------------------
# Configuracion de pagina + paleta institucional navy/gold
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Ernesto Investing AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#0B1F3A"
NAVY_LIGHT = "#13294B"
GOLD = "#C9A227"
GOLD_LIGHT = "#E4C767"
GREEN = "#2E7D32"
RED = "#B3261E"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #F4F6F9; }}
    section[data-testid="stSidebar"] {{ background-color: {NAVY}; }}
    section[data-testid="stSidebar"] * {{ color: #F4F6F9 !important; }}
    h1, h2, h3 {{ color: {NAVY}; }}
    div[data-testid="stMetric"] {{
        background-color: white;
        border: 1px solid #E0E4EA;
        border-left: 4px solid {GOLD};
        border-radius: 8px;
        padding: 10px 14px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Conexion a MongoDB Atlas (usa Secrets de Streamlit Cloud)
# ------------------------------------------------------------------
TICKERS = {
    "FSM": "Fortuna Silver Mines Inc.",
    "VOLCABC1.LM": "Volcan Compania Minera S.A.A.",
    "ABX.TO": "Barrick Gold Corporation",
    "BVN": "Compania de Minas Buenaventura",
    "BHP": "BHP Billiton Limited",
}


@st.cache_resource(show_spinner=False)
def conectar_mongo():
    """Crea el cliente de MongoDB usando la URI guardada en st.secrets."""
    mongo_uri = st.secrets["MONGO_URI"]
    client = MongoClient(mongo_uri, server_api=ServerApi("1"))
    client.admin.command("ping")
    return client["ernesto_investing_ai"]


try:
    db = conectar_mongo()
    conexion_ok = True
except Exception as e:
    conexion_ok = False
    error_conexion = str(e)


@st.cache_data(ttl=300, show_spinner=False)
def cargar_precios(ticker):
    docs = list(
        db["precios_ohlcv"].find({"ticker": ticker}, {"_id": 0, "created_at": 0}).sort("fecha", 1)
    )
    return pd.DataFrame(docs)


@st.cache_data(ttl=300, show_spinner=False)
def cargar_prediccion(ticker):
    return db["predicciones"].find_one(
        {"ticker": ticker, "modelo": "SVC"}, {"_id": 0, "created_at": 0}
    )


@st.cache_data(ttl=300, show_spinner=False)
def cargar_metricas(ticker):
    return db["metricas_modelos"].find_one(
        {"ticker": ticker, "modelo": "SVC"}, {"_id": 0, "fecha_entrenamiento": 0}
    )


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.title("📈 Ernesto Investing AI")
st.sidebar.caption("Sistema de Soporte a la Decision de Inversion")
st.sidebar.markdown("---")

ticker = st.sidebar.selectbox(
    "Selecciona un activo minero",
    options=list(TICKERS.keys()),
    format_func=lambda t: f"{t} — {TICKERS[t]}",
)

st.sidebar.markdown("---")
if conexion_ok:
    st.sidebar.success("MongoDB Atlas: conectado")
else:
    st.sidebar.error("MongoDB Atlas: sin conexion")

st.sidebar.caption(
    "Bono Streamlit · URL permanente\n"
    "No depende de que los notebooks de Colab esten ejecutandose."
)

st.title("Ernesto Investing AI")
st.caption("Decision Support System para inversion en mineras con operaciones en Peru")

if not conexion_ok:
    st.error(
        "No se pudo conectar a MongoDB Atlas. Verifica el secret **MONGO_URI** "
        "en *Advanced settings → Secrets* de Streamlit Cloud."
    )
    st.code(error_conexion)
    st.stop()

tab_mercado, tab_svc = st.tabs(["📊 Market Dashboard", "🤖 SVC Classifier"])

# ------------------------------------------------------------------
# TAB 1 — Market Dashboard (equivalente a modulo_mercado.html)
# ------------------------------------------------------------------
with tab_mercado:
    df = cargar_precios(ticker)

    if df.empty:
        st.warning(f"No hay datos en MongoDB para {ticker}. Ejecuta el Notebook 1.")
    else:
        df["fecha"] = pd.to_datetime(df["fecha"])
        ultimo = df.iloc[-1]
        anterior = df.iloc[-2] if len(df) > 1 else ultimo
        variacion = ultimo["close"] - anterior["close"]
        variacion_pct = (variacion / anterior["close"] * 100) if anterior["close"] else 0

        st.subheader(f"{ticker} — {TICKERS[ticker]}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ultimo cierre", f"$ {ultimo['close']:.2f}", f"{variacion_pct:+.2f}%")
        c2.metric("SMA 20", f"$ {ultimo['sma_20']:.2f}" if pd.notna(ultimo["sma_20"]) else "—")
        c3.metric("RSI 14", f"{ultimo['rsi_14']:.1f}" if pd.notna(ultimo["rsi_14"]) else "—")
        c4.metric("Volumen", f"{int(ultimo['volume']):,}")

        # Grafico candlestick + medias moviles + RSI (subplot inferior)
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
            vertical_spacing=0.04,
            subplot_titles=("Precio OHLC + Medias Moviles", "RSI (14)"),
        )

        fig.add_trace(
            go.Candlestick(
                x=df["fecha"], open=df["open"], high=df["high"],
                low=df["low"], close=df["close"], name="OHLC",
                increasing_line_color=GREEN, decreasing_line_color=RED,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df["fecha"], y=df["sma_20"], name="SMA 20",
                       line=dict(color=GOLD, width=1.5)),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df["fecha"], y=df["sma_50"], name="SMA 50",
                       line=dict(color=NAVY, width=1.5)),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df["fecha"], y=df["ema_12"], name="EMA 12",
                       line=dict(color=GOLD_LIGHT, width=1, dash="dot")),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(x=df["fecha"], y=df["rsi_14"], name="RSI 14",
                       line=dict(color=NAVY_LIGHT, width=1.5)),
            row=2, col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color=RED, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=GREEN, row=2, col=1)

        fig.update_layout(
            height=650, xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=60, b=20, l=10, r=10),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        fig.update_yaxes(range=[0, 100], row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Ver datos crudos (precios_ohlcv)"):
            st.dataframe(
                df[["fecha", "open", "high", "low", "close", "volume",
                    "sma_20", "sma_50", "ema_12", "ema_26", "rsi_14"]]
                .sort_values("fecha", ascending=False),
                use_container_width=True,
            )

# ------------------------------------------------------------------
# TAB 2 — SVC Classifier (equivalente a modulo_svc.html)
# ------------------------------------------------------------------
with tab_svc:
    prediccion = cargar_prediccion(ticker)
    metricas = cargar_metricas(ticker)

    if not prediccion:
        st.warning(f"No hay prediccion SVC para {ticker}. Ejecuta el Notebook 2.")
    else:
        st.subheader(f"Señal del clasificador SVC — {ticker}")

        senal = prediccion["senal"]
        confianza = prediccion["confianza"]
        color_senal = GREEN if senal == "BUY" else RED
        icono = "🟢" if senal == "BUY" else "🔴"

        col_senal, col_conf, col_fecha = st.columns(3)
        with col_senal:
            st.markdown(
                f"""<div style="background-color:{color_senal};color:white;
                border-radius:10px;padding:22px;text-align:center;">
                <div style="font-size:14px;opacity:.85;">SEÑAL ACTUAL</div>
                <div style="font-size:34px;font-weight:700;">{icono} {senal}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        col_conf.metric("Confianza del modelo", f"{confianza:.0%}")
        col_fecha.metric("Fecha de prediccion", prediccion.get("fecha_prediccion", "—"))

        st.markdown("---")

        if metricas:
            st.markdown("#### Metricas de entrenamiento (particion temporal 80/20)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metricas['accuracy']:.0%}")
            m2.metric("Precision", f"{metricas['precision']:.0%}")
            m3.metric("Recall", f"{metricas['recall']:.0%}")
            m4.metric("F1-Score", f"{metricas['f1']:.0%}")

            st.caption(
                f"Mejor kernel: **{metricas['mejor_kernel']}** · "
                f"C = **{metricas['mejor_C']}** · gamma = **{metricas['mejor_gamma']}**"
            )

            cm = metricas.get("matriz_confusion")
            if cm:
                col_cm, col_info = st.columns([1, 1])
                with col_cm:
                    fig_cm = go.Figure(
                        data=go.Heatmap(
                            z=cm,
                            x=["Pred: SELL", "Pred: BUY"],
                            y=["Real: SELL", "Real: BUY"],
                            colorscale=[[0, "#F4F6F9"], [1, NAVY]],
                            text=cm, texttemplate="%{text}",
                            showscale=False,
                        )
                    )
                    fig_cm.update_layout(
                        title="Matriz de confusion",
                        height=350, margin=dict(t=40, b=10, l=10, r=10),
                        yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                with col_info:
                    st.info(
                        "**Como leer la matriz:** las filas son la señal real y las "
                        "columnas la señal predicha por el SVC. La diagonal principal "
                        "son los aciertos del modelo; fuera de la diagonal son los errores."
                    )
        else:
            st.info("Aun no hay metricas de entrenamiento guardadas para este ticker.")

st.markdown("---")
st.caption(
    "Ernesto Investing AI · iDeSo · UNMSM · Prof. E. D. Cancho-Rodríguez, MBA (GWU) · "
    "Datos leidos en tiempo real desde MongoDB Atlas (sin recalcular)."
)
