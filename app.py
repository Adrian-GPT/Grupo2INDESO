# app.py — Ernesto Investing AI (Bono Streamlit, Semana 14)
# ------------------------------------------------------------------
# App 100% Python que lee datos YA CALCULADOS de MongoDB Atlas
# (no recalcula nada: mismas colecciones pobladas por los 6 notebooks
# del backend) y los muestra con st.plotly_chart, st.metric y
# st.selectbox.
#
# Colecciones leidas:
#   Notebook 1 (Ingesta)     -> precios_ohlcv
#   Notebook 2 (SVC)         -> predicciones (modelo=SVC), metricas_modelos (modelo=SVC)
#   Notebook 3 (RNN)         -> predicciones (modelo en LSTM/BiLSTM/GRU/SimpleRNN),
#                                metricas_modelos (idem, incluye historial_entrenamiento)
#   Notebook 4 (Backtest)    -> resultados_backtest
#   Notebook 5 (LSTM Regres) -> predicciones_lstm, metricas_lstm
#   Notebook 6 (NLP/VADER)   -> sentimiento_nlp
# ------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
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
BLUE = "#3B82F6"
GRAY = "#6B7A99"

RNN_COLORS = {"LSTM": GOLD, "BiLSTM": BLUE, "GRU": GREEN, "SimpleRNN": GRAY}

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

RNN_MODELOS = ["LSTM", "BiLSTM", "GRU", "SimpleRNN"]
HORIZONTES_LSTM = [7, 14, 30, 60]


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


# ------------------------------------------------------------------
# Funciones de carga (cacheadas 5 min) — una por coleccion
# ------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def cargar_precios(ticker):
    docs = list(
        db["precios_ohlcv"].find({"ticker": ticker}, {"_id": 0, "created_at": 0}).sort("fecha", 1)
    )
    return pd.DataFrame(docs)


@st.cache_data(ttl=300, show_spinner=False)
def cargar_prediccion(ticker, modelo):
    return db["predicciones"].find_one(
        {"ticker": ticker, "modelo": modelo}, {"_id": 0, "created_at": 0}
    )


@st.cache_data(ttl=300, show_spinner=False)
def cargar_metricas(ticker, modelo):
    return db["metricas_modelos"].find_one(
        {"ticker": ticker, "modelo": modelo}, {"_id": 0, "fecha_entrenamiento": 0}
    )


@st.cache_data(ttl=300, show_spinner=False)
def cargar_backtest(ticker):
    return db["resultados_backtest"].find_one(
        {"ticker": ticker, "estrategia": "SMA_20_50"}, {"_id": 0}
    )


@st.cache_data(ttl=300, show_spinner=False)
def cargar_prediccion_lstm(ticker):
    return db["predicciones_lstm"].find_one({"ticker": ticker}, {"_id": 0})


@st.cache_data(ttl=300, show_spinner=False)
def cargar_metricas_lstm(ticker):
    return db["metricas_lstm"].find_one(
        {"ticker": ticker}, {"_id": 0, "fecha_entrenamiento": 0}
    )


@st.cache_data(ttl=300, show_spinner=False)
def cargar_sentimiento(ticker):
    return db["sentimiento_nlp"].find_one({"ticker": ticker}, {"_id": 0})


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
st.sidebar.markdown("---")
st.sidebar.caption(
    "Modulos: Market Dashboard, SVC, RNN (4 arquitecturas), "
    "LSTM Regressor, Backtesting y NLP/VADER."
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

tab_mercado, tab_svc, tab_rnn, tab_lstm, tab_backtest, tab_nlp = st.tabs(
    ["📊 Market Dashboard", "🤖 SVC Classifier", "🔁 RNN (4 arquitecturas)",
     "📈 LSTM Regressor", "💰 Backtesting", "📰 NLP / VADER"]
)

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
    prediccion = cargar_prediccion(ticker, "SVC")
    metricas = cargar_metricas(ticker, "SVC")

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

# ------------------------------------------------------------------
# TAB 3 — RNN: 4 arquitecturas (LSTM, BiLSTM, GRU, SimpleRNN)
# equivalente a investai_modelos_ia.html — Notebook 3
# ------------------------------------------------------------------
with tab_rnn:
    st.subheader(f"Clasificadores recurrentes — {ticker}")
    st.caption(
        "Comparacion de 4 arquitecturas entrenadas con la misma semilla y "
        "el mismo conjunto de features que el SVC, para predecir la señal BUY/SELL."
    )

    filas_resumen = []
    predicciones_rnn = {}
    metricas_rnn = {}
    for modelo_nombre in RNN_MODELOS:
        pred = cargar_prediccion(ticker, modelo_nombre)
        met = cargar_metricas(ticker, modelo_nombre)
        predicciones_rnn[modelo_nombre] = pred
        metricas_rnn[modelo_nombre] = met
        if pred and met:
            filas_resumen.append({
                "Arquitectura": modelo_nombre,
                "Señal": pred["senal"],
                "Confianza": f"{pred['confianza']:.0%}",
                "Accuracy": f"{met['accuracy']:.0%}",
                "F1-Score": f"{met['f1']:.0%}",
            })

    if not filas_resumen:
        st.warning(f"No hay predicciones RNN para {ticker}. Ejecuta el Notebook 3.")
    else:
        st.markdown("#### Resumen comparativo de las 4 arquitecturas")
        st.dataframe(pd.DataFrame(filas_resumen), use_container_width=True, hide_index=True)

        st.markdown("---")
        modelo_sel = st.radio(
            "Explorar arquitectura en detalle", RNN_MODELOS, horizontal=True, key="rnn_radio"
        )
        pred = predicciones_rnn.get(modelo_sel)
        met = metricas_rnn.get(modelo_sel)

        if pred and met:
            senal = pred["senal"]
            color_senal = GREEN if senal == "BUY" else RED
            icono = "🟢" if senal == "BUY" else "🔴"

            col_senal, col_conf, col_acc = st.columns(3)
            with col_senal:
                st.markdown(
                    f"""<div style="background-color:{color_senal};color:white;
                    border-radius:10px;padding:22px;text-align:center;">
                    <div style="font-size:14px;opacity:.85;">SEÑAL — {modelo_sel}</div>
                    <div style="font-size:34px;font-weight:700;">{icono} {senal}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            col_conf.metric("Confianza", f"{pred['confianza']:.0%}")
            col_acc.metric("Accuracy (test)", f"{met['accuracy']:.0%}")

            st.markdown("#### Curva de entrenamiento (loss y accuracy por epoca)")
            hist = met.get("historial_entrenamiento")
            if hist:
                fig_hist = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"))
                epocas = list(range(1, len(hist["loss"]) + 1))
                fig_hist.add_trace(go.Scatter(x=epocas, y=hist["loss"], name="Train loss",
                                               line=dict(color=NAVY)), row=1, col=1)
                fig_hist.add_trace(go.Scatter(x=epocas, y=hist["val_loss"], name="Val loss",
                                               line=dict(color=RED, dash="dot")), row=1, col=1)
                fig_hist.add_trace(go.Scatter(x=epocas, y=hist["accuracy"], name="Train acc",
                                               line=dict(color=GOLD)), row=1, col=2)
                fig_hist.add_trace(go.Scatter(x=epocas, y=hist["val_accuracy"], name="Val acc",
                                               line=dict(color=GREEN, dash="dot")), row=1, col=2)
                fig_hist.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white",
                                        margin=dict(t=50, b=10, l=10, r=10))
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("Este registro no tiene historial de entrenamiento guardado.")

            cm = met.get("matriz_confusion")
            if cm:
                fig_cm = go.Figure(
                    data=go.Heatmap(
                        z=cm, x=["Pred: SELL", "Pred: BUY"], y=["Real: SELL", "Real: BUY"],
                        colorscale=[[0, "#F4F6F9"], [1, RNN_COLORS.get(modelo_sel, NAVY)]],
                        text=cm, texttemplate="%{text}", showscale=False,
                    )
                )
                fig_cm.update_layout(title=f"Matriz de confusion — {modelo_sel}",
                                      height=320, margin=dict(t=40, b=10, l=10, r=10),
                                      yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info(f"No hay datos guardados para la arquitectura {modelo_sel} en {ticker}.")

# ------------------------------------------------------------------
# TAB 4 — LSTM Regressor: pronostico de precio multi-horizonte
# equivalente a investai_lstm.html — Notebook 5
# ------------------------------------------------------------------
with tab_lstm:
    pred_lstm = cargar_prediccion_lstm(ticker)
    met_lstm = cargar_metricas_lstm(ticker)

    if not pred_lstm:
        st.warning(f"No hay pronostico LSTM para {ticker}. Ejecuta el Notebook 5.")
    else:
        st.subheader(f"Pronostico de precio (LSTM Regressor) — {ticker}")
        st.caption(f"Fecha base del pronostico: {pred_lstm.get('fecha_base', '—')}")

        if met_lstm:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("RMSE", f"$ {met_lstm['rmse_usd']:.2f}", f"{met_lstm['rmse_pct']:.1f}%")
            m2.metric("MAE", f"$ {met_lstm['mae_usd']:.2f}")
            m3.metric("R²", f"{met_lstm['r2']:.3f}")
            mejora = met_lstm.get("mejora_vs_arima_pct")
            m4.metric("Mejora vs ARIMA(1,1,1)", f"{mejora:+.1f}%" if mejora is not None else "—")

        horizonte_sel = st.select_slider(
            "Horizonte de pronostico (dias)", options=HORIZONTES_LSTM, value=30
        )

        horizontes = pred_lstm.get("horizontes", {})
        datos_h = horizontes.get(str(horizonte_sel))

        if not datos_h:
            st.info(f"No hay pronostico guardado para el horizonte de {horizonte_sel} dias.")
        else:
            df_h = pd.DataFrame(datos_h)
            df_h["fecha_proyectada"] = pd.to_datetime(df_h["fecha_proyectada"])

            df_precios = cargar_precios(ticker)
            fig = go.Figure()

            if not df_precios.empty:
                df_precios["fecha"] = pd.to_datetime(df_precios["fecha"])
                df_hist = df_precios.tail(60)
                fig.add_trace(go.Scatter(
                    x=df_hist["fecha"], y=df_hist["close"], name="Historico (60d)",
                    line=dict(color=NAVY, width=1.5),
                ))

            fig.add_trace(go.Scatter(
                x=df_h["fecha_proyectada"], y=df_h["banda_superior"], name="Banda superior",
                line=dict(color=GOLD_LIGHT, width=0), showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=df_h["fecha_proyectada"], y=df_h["banda_inferior"], name="Banda de confianza",
                line=dict(color=GOLD_LIGHT, width=0), fill="tonexty",
                fillcolor="rgba(228,199,103,0.3)",
            ))
            fig.add_trace(go.Scatter(
                x=df_h["fecha_proyectada"], y=df_h["precio_proyectado_usd"], name="Pronostico",
                line=dict(color=GOLD, width=2.5, dash="dash"),
            ))

            fig.update_layout(
                height=480, plot_bgcolor="white", paper_bgcolor="white",
                title=f"Pronostico a {horizonte_sel} dias",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=60, b=20, l=10, r=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Ver tabla del pronostico"):
                st.dataframe(
                    df_h[["fecha_proyectada", "precio_proyectado_usd",
                          "banda_inferior", "banda_superior"]],
                    use_container_width=True, hide_index=True,
                )

# ------------------------------------------------------------------
# TAB 5 — Backtesting: estrategia SMA 20/50
# equivalente a reportes_backtesting.html — Notebook 4
# ------------------------------------------------------------------
with tab_backtest:
    bt = cargar_backtest(ticker)

    if not bt:
        st.warning(f"No hay backtest para {ticker}. Ejecuta el Notebook 4.")
    else:
        st.subheader(f"Backtest — Estrategia cruce SMA 20/50 — {ticker}")

        m1, m2, m3, m4 = st.columns(4)
        rt = bt.get("retorno_total")
        m1.metric("Retorno total", f"{rt:+.2%}" if rt is not None else "—")
        sh = bt.get("sharpe_ratio")
        m2.metric("Sharpe ratio", f"{sh:.2f}" if sh is not None else "—")
        dd = bt.get("max_drawdown")
        m3.metric("Max drawdown", f"{dd:.2%}" if dd is not None else "—")
        wr = bt.get("win_rate")
        m4.metric("Win rate", f"{wr:.0%}" if wr is not None else "—",
                   f"{bt.get('num_operaciones', 0)} operaciones")

        col_cap1, col_cap2 = st.columns(2)
        col_cap1.metric("Capital inicial", f"$ {bt.get('capital_inicial', 0):,.2f}")
        col_cap2.metric("Capital final", f"$ {bt.get('capital_final', 0):,.2f}")

        curva = bt.get("curva_equity", [])
        if curva:
            df_curva = pd.DataFrame(curva)
            df_curva["fecha"] = pd.to_datetime(df_curva["fecha"])
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=df_curva["fecha"], y=df_curva["valor"], name="Equity",
                line=dict(color=GOLD, width=2), fill="tozeroy",
                fillcolor="rgba(201,162,39,0.12)",
            ))
            fig_eq.add_hline(y=bt.get("capital_inicial", 0), line_dash="dot", line_color=GRAY)
            fig_eq.update_layout(
                height=420, plot_bgcolor="white", paper_bgcolor="white",
                title="Curva de equity", margin=dict(t=50, b=20, l=10, r=10),
            )
            st.plotly_chart(fig_eq, use_container_width=True)

        operaciones = bt.get("operaciones", [])
        if operaciones:
            st.markdown("#### Operaciones cerradas")
            df_ops = pd.DataFrame(operaciones)
            st.dataframe(df_ops, use_container_width=True, hide_index=True)
        else:
            st.info("No se registraron operaciones cerradas en el periodo evaluado.")

# ------------------------------------------------------------------
# TAB 6 — NLP / VADER: sentimiento de noticias
# equivalente a investai_nlp.html — Notebook 6
# ------------------------------------------------------------------
with tab_nlp:
    sent = cargar_sentimiento(ticker)

    if not sent:
        st.warning(f"No hay analisis de sentimiento para {ticker}. Ejecuta el Notebook 6.")
    else:
        st.subheader(f"Sentimiento de noticias (VADER) — {ticker}")

        score = sent.get("score_promedio", 0.0)
        clasif = sent.get("clasificacion_general", "—")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number={"valueformat": ".3f"},
            title={"text": f"Score promedio — {clasif}"},
            gauge={
                "axis": {"range": [-1, 1]},
                "bar": {"color": NAVY},
                "steps": [
                    {"range": [-1, -0.05], "color": "#F8D7D5"},
                    {"range": [-0.05, 0.05], "color": "#FBF0CE"},
                    {"range": [0.05, 1], "color": "#D7ECD9"},
                ],
                "threshold": {"line": {"color": GOLD, "width": 3},
                              "thickness": 0.8, "value": score},
            },
        ))
        fig_gauge.update_layout(height=320, margin=dict(t=60, b=10, l=30, r=30),
                                 paper_bgcolor="white")
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.metric("Noticias analizadas", sent.get("cantidad_noticias", 0))

        noticias = sent.get("noticias", [])
        if noticias:
            st.markdown("#### Titulares analizados")
            for n in noticias:
                clasificacion_n = n.get("clasificacion", "neutral")
                color_n = GREEN if clasificacion_n == "positivo" else (
                    RED if clasificacion_n == "negativo" else GRAY)
                compound = n.get("scores", {}).get("compound", 0.0)
                st.markdown(
                    f"""<div style="border-left:4px solid {color_n};background:white;
                    border-radius:6px;padding:10px 14px;margin-bottom:8px;">
                    <b>{n.get('titulo', '')}</b><br>
                    <span style="color:{GRAY};font-size:12px;">
                    {n.get('publicador', '')} · {n.get('fecha_publicacion', '')} ·
                    compound: {compound:+.3f} ({clasificacion_n})</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No hay titulares individuales guardados para este ticker.")

st.markdown("---")
st.caption(
    "Ernesto Investing AI · iDeSo · UNMSM · Prof. E. D. Cancho-Rodríguez, MBA (GWU) · "
    "Datos leidos en tiempo real desde MongoDB Atlas (sin recalcular)."
)
