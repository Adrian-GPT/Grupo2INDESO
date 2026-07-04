[README.md](https://github.com/user-attachments/files/29651986/README.md)
# 📈 Ernesto Investing AI
**Sistema Operacional Integrado: Frontend + Backend (FastAPI) + MongoDB**

![UNMSM](https://img.shields.io/badge/UNMSM-FISI-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-00a393)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248)
![scikit-learn](https://img.shields.io/badge/scikit--learn-SVC-F7931E)

Este repositorio contiene la implementación final del proyecto **Ernesto Investing AI**, desarrollado para la Semana 13 del curso **Introducción al Desarrollo de Software (iDeSo)** de la Facultad de Ingeniería de Sistemas e Informática (FISI) de la Universidad Nacional Mayor de San Marcos (UNMSM).

**Profesor:** Mg. Ing. Ernesto D. Cancho-Rodríguez, MBA

## 📋 Descripción del Proyecto

El sistema es una plataforma completa de análisis de mercado financiero que se enfoca en 5 tickers del sector minero. Descarga datos reales, calcula indicadores técnicos (SMA, EMA, RSI), entrena un modelo predictivo de Machine Learning (SVC) para generar señales de compra/venta (BUY/SELL), y expone todo esto a través de una API RESTful conectada a un Frontend interactivo.

## 🏗 Arquitectura y Fases del Sistema


### 1. Ingesta de Datos y Machine Learning (`Notebooks 1 y 2`)
* **Notebook 1 (Ingesta de Datos):** Extrae datos de la bolsa mediante `yfinance` para los 5 tickers mineros, calcula indicadores técnicos y almacena el histórico (OHLCV) en una base de datos de **MongoDB Atlas**.
* **Notebook 2 (Clasificador SVC):** Lee los datos de MongoDB, entrena un modelo *Support Vector Classifier* usando `GridSearchCV` y guarda las predicciones y métricas de evaluación en nuevas colecciones en la nube (`predicciones` y `metricas_modelos`).

### 2. Backend y API REST (`Notebook 3`)
* **Notebook 3 (API FastAPI):** Levanta un servidor web usando **FastAPI** en Google Colab, que lee los datos ya procesados en MongoDB y los expone a Internet de forma pública mediante **ngrok**. 
* **Endpoints principales:**
  * `GET /api/salud`: Verificación de estado del servidor y conexión a DB.
  * `GET /api/mercado/{ticker}`: Retorna datos de mercado e indicadores técnicos.
  * `GET /api/svc/{ticker}`: Retorna las predicciones, señales y métricas del modelo ML.

### 3. Frontend Web (`HTML/JS/CSS`)
* Interfaz web lista para ser desplegada en **GitHub Pages**.
* Consume dinámicamente los datos proporcionados por la API expuesta a través de ngrok. Se compone de una página de configuración (`index.html`), un dashboard de mercado (`modulo_mercado.html`) y el módulo (`modulo_svc.html`) que interactúan con el backend.

## 🚀 Instrucciones de Despliegue

Para poner en marcha el sistema completo, sigue estos pasos estrictamente en orden:

### Backend (Google Colab + MongoDB)
1. Abrir **Notebook 1** en Google Colab, configurar los *Secrets* (`MONGO_URI`) y ejecutar todo para poblar la colección `precios_ohlcv` en MongoDB.
2. Abrir **Notebook 2** en Google Colab y ejecutar todo para entrenar el modelo SVC y guardar los resultados.
3. Abrir **Notebook 3** en Google Colab, configurar los *Secrets* (`MONGO_URI` y `NGROK_AUTHTOKEN`) y levantar el servidor. Al final de la ejecución, **copia la URL pública de ngrok** generada.

### Frontend
1. Abre el archivo `index.html` del repositorio desde tu navegador.
2. En el panel de configuración, pega la **URL de ngrok** obtenida en el Paso 3 del backend.
3. Navega hacia los módulos para visualizar los tableros analíticos con datos reales provenientes de la API.

*(Nota: Dado el uso del nivel gratuito de ngrok, es normal que la API retorne una advertencia en el navegador en la primera visita. El frontend usa el header `ngrok-skip-browser-warning` para evitar bloqueos en las peticiones de fetch).*

## 🛠 Tecnologías Utilizadas
* **Backend:** Python, FastAPI, uvicorn
* **Machine Learning / Data:** scikit-learn, yfinance, pandas
* **Base de Datos:** MongoDB Atlas, pymongo
* **Infraestructura:** Google Colab, ngrok
* **Frontend:** HTML5, CSS3, Vanilla JavaScript (Fetch API)

## 📄 Licencia
Proyecto académico elaborado para el curso iDeSo - UNMSM FISI.
