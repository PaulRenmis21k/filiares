# 🥔 Potato Disease Classifier

> **Proyecto Universitario · Primera Unidad**
> Aplicación con elemento de predicción mediante aprendizaje supervisado en un ciclo de vida regular de software.

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Dataset](#-dataset)
- [Modelo de Machine Learning](#-modelo-de-machine-learning)
- [Aplicación Web](#-aplicación-web)
- [Despliegue con Docker](#-despliegue-con-docker)
- [CI/CD con GitHub Actions](#-cicd-con-github-actions)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación y Uso](#-instalación-y-uso)
- [Pruebas](#-pruebas)
- [Mantenimiento](#-mantenimiento)

---

## 📖 Descripción del Proyecto

Este proyecto implementa una **aplicación web con un modelo de aprendizaje supervisado** (CNN) para clasificar enfermedades en hojas de papa. El sistema está diseñado siguiendo un **ciclo de vida regular de software**, incluyendo:

- **Modelo ML**: Red neuronal convolucional entrenada con TensorFlow/Keras
- **API REST**: Backend con FastAPI
- **Frontend**: Interfaz web moderna con upload de imágenes y resultados en tiempo real
- **Contenedorización**: Docker para despliegue reproducible
- **CI/CD**: Pipelines automatizados con GitHub Actions (tests, build, deploy, retrain)
- **Mantenimiento**: Pipeline semanal de re-entrenamiento automatizado

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    Usuario (Browser)                         │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP (HTML/JSON)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Server (port 8000)                 │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │ Templates   │  │ Static Files │  │ REST API         │    │
│  │ (Jinja2)    │  │ (CSS, JS)    │  │ /predict, /health │    │
│  └─────────────┘  └──────────────┘  └────────┬─────────┘    │
└───────────────────────────────────────────────┬──────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────┐
│                Model Predictor (TensorFlow/Keras)            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │    potato_disease_model.keras (CNN, 3 clases)        │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Flujo de CI/CD

```
[Commit] → [Tests Automatizados] → [Build Docker] → [Push Registry] → [Deploy]
                                                                        ↓
[Schedule Semanal] → [Re-entrenamiento] → [Commit Modelo] → [Re-deploy]
```

## 📊 Dataset

**Dataset**: [Potato Dataset](https://www.kaggle.com/datasets/arajmishra/potato-dataset) (vía KaggleHub)

- **Fuente**: PlantVillage - Subconjunto de hojas de papa
- **Total de imágenes**: ~2,150
- **Clases**: 3 categorías

| Clase | Enfermedad | Agente Causal | Severidad |
|-------|-----------|---------------|-----------|
| **Early Blight** | Tizón temprano | *Alternaria solani* | Moderado |
| **Late Blight** | Tizón tardío | *Phytophthora infestans* | Severo |
| **Healthy** | Saludable | - | Ninguna |

## 🧠 Modelo de Machine Learning

### Arquitectura: MobileNetV2 (Transfer Learning)

```
Input: (128, 128, 3) RGB Image (normalized to [-1, 1])
    │
    ├── MobileNetV2 Base (pre-trained on ImageNet, frozen)
    │   ├── Inverted Residual Blocks (x17)
    │   ├── Global Average Pooling (pooling='avg')
    │   └── Output: 1280 feature vector
    │
    ├── Dropout(0.5)
    ├── Dense(3, Softmax)
    │   ├── Early Blight
    │   ├── Late Blight
    │   └── Healthy
```

> **Nota**: La aumentación de datos (data augmentation) se aplica en el pipeline de `tf.data` (RandomFlip, Rot90, Brightness, Contrast), no dentro del modelo.

### Hiperparámetros

| Parámetro | Valor |
|-----------|-------|
| Base Model | MobileNetV2 (ImageNet weights, frozen) |
| Optimizer | Adam (learning_rate=0.001) |
| Loss | Categorical Crossentropy |
| Batch Size | 32 |
| Input Size | 128×128×3 |
| Epochs | 5 (con Early Stopping patience=4) |
| Reduce LR Patience | 2 épocas |
| Reduce LR Factor | 0.5 |
| Train/Val/Test Split | 70%/20%/10% |
| Data Augmentation | Sí (Flip, Rotation, Brightness, Contrast) - vía tf.data |

### Métricas Obtenidas

- **Accuracy (Test)**: **97.02%**
- **Loss (Test)**: 0.1041
- **Best Val Accuracy**: 97.36%
- **Parámetros Totales**: ~2.5M (MobileNetV2 frozen + top dense)

## 🌐 Aplicación Web

### Tecnologías

- **Backend**: Python 3.13, FastAPI, Uvicorn
- **Frontend**: HTML, CSS (modern dark theme), JavaScript vanilla
- **ML**: TensorFlow 2.20, Keras 3.12

### Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Página principal con upload |
| GET | `/health` | Health check para monitoreo |
| POST | `/predict` | Clasificar imagen (multipart upload) |
| GET | `/api/model-info` | Información del modelo |
| GET | `/api/classes` | Lista de clases disponibles |

## 🐳 Despliegue con Docker (Local)

### Construir imagen

```bash
docker build -t potato-disease-classifier .
```

### Ejecutar con docker-compose

```bash
docker-compose up -d
```

La aplicación estará disponible en `http://localhost:8000`.

### Verificar estado

```bash
curl http://localhost:8000/health
```

## ☁️ Despliegue en Render.com (Producción)

[Render](https://render.com) es una plataforma cloud gratuita ideal para proyectos universitarios.
La aplicación se despliega automáticamente desde GitHub.

### Paso 1: Subir a GitHub

```bash
# Crear el repositorio en GitHub y subir el código
git init
git add .
git commit -m "Potato Disease Classifier - v1.0"
git remote add origin https://github.com/tu-usuario/potato-disease-classifier.git
git push -u origin main
```

### Paso 2: Crear Web Service en Render

1. Ve a [https://dashboard.render.com](https://dashboard.render.com) y crea una cuenta (gratis)
2. Haz clic en **"New +"** → **"Web Service"**
3. Conecta tu repositorio de GitHub
4. Render **detectará automáticamente** el archivo `render.yaml` y lo configurará solo
   - Si no detecta el blueprint, selecciona:
     - **Runtime**: `Python`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
     - **Health Check Path**: `/health`
5. Selecciona el plan **Free** ($0/mes)
6. Haz clic en **"Deploy Web Service"**

### Paso 3: ¡App en producción!

Render te dará una URL como:
```
https://potato-disease-classifier.onrender.com
```

La aplicación estará disponible en esa URL para que:
- Cualquier persona pueda subir imágenes y clasificarlas
- Los usuarios puedan confirmar/corregir predicciones (feedback)
- El pipeline de CI/CD se active con cada push

### Mantenimiento en Render

Render ofrece:
- **Health checks automáticos** (endpoint `/health`)
- **Logs en tiempo real** desde el dashboard
- **Reinicio automático** si la app falla
- **0$ de costo** en el plan free (750 horas/mes)

> ⚠️ **Nota**: En el plan free, el servicio se "duerme" después de 15 min sin actividad.
> La primera solicitud después del sleep tarda ~30s en responder (cold start).
> Esto no afecta la evaluación del proyecto, pero tenlo en cuenta en la demo.

## 🔄 CI/CD con GitHub Actions

### Workflows

El pipeline `ci-cd.yml` incluye 3 jobs principales:

#### 1. `test` (se ejecuta en cada push y PR)
- Checkout del código
- Setup de Python 3.13 con caché
- Instalación de dependencias
- Linting con flake8
- Verificación de formato con black
- Tests con pytest y reporte de covertura

#### 2. `build-and-push` (se ejecuta en push a main/master)
- Build de imagen Docker multi-stage
- Push a GitHub Container Registry (GHCR)
- Cache de Docker layers para builds rápidos

#### 3. `retrain-model` (se ejecuta semanalmente o manualmente)
- Re-entrenamiento del modelo con datos actualizados
- Subida del modelo como artifact
- Commit automático del nuevo modelo al repositorio

### Activación manual

Para re-entrenar manualmente el modelo desde GitHub Actions:

1. Ir a Actions → CI/CD Pipeline
2. Click "Run workflow"
3. Marcar "Retrain model" como `true`
4. Ejecutar

## 📁 Estructura del Proyecto

```
potato-disease-classifier/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application
│   ├── model/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Training script (CNN architecture)
│   │   └── predictor.py         # Prediction module (singleton)
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css        # Dark theme UI
│   │   └── js/
│   │       └── main.js          # Upload & classification UI logic
│   └── templates/
│       ├── index.html           # Main page
│       └── result.html          # Result page
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_app.py              # 15+ tests
├── notebooks/
│   └── model_training.ipynb     # Jupyter notebook for training
├── models/                      # Saved models (gitignored)
├── uploads/                     # Uploaded images (gitignored)
├── training_data/               # User feedback images for retraining
│   ├── early_blight/
│   ├── late_blight/
│   └── healthy/
├── .github/
│   └── workflows/
│       └── ci-cd.yml            # GitHub Actions pipeline
├── .gitignore
├── requirements.txt
├── Dockerfile                   # Multi-stage build
├── docker-compose.yml
└── README.md                    # This file
```

## 🚀 Instalación y Uso

### Requisitos

- Python 3.10+
- TensorFlow 2.10+ (con GPU opcional)
- Docker (opcional, para despliegue)

### Instalación local

```bash
# 1. Clonar el repositorio
git clone <repo-url>
cd potato-disease-classifier

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Entrenar el modelo
python -m app.model.trainer

# 5. Iniciar la aplicación
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Acceder a la aplicación

Abrir en el navegador: **http://localhost:8000**

## 🧪 Pruebas

### Ejecutar tests

```bash
pytest tests/ -v
```

### Con covertura

```bash
pytest tests/ -v --cov=app/
```

### Tests incluidos

- **Health check**: Verificar estado del servidor
- **Home page**: Renderizado correcto
- **Predicción**: Sin modelo (503), con archivos inválidos
- **API endpoints**: `/api/classes`, `/api/model-info`
- **Preprocesamiento**: Imágenes RGB, RGBA, escala de grises
- **Arquitectura**: Build del modelo, capas de augmentación
- **Archivos estáticos**: CSS y JS accesibles

## 🔧 Mantenimiento

### Re-entrenamiento automático

El modelo se re-entrena automáticamente cada semana mediante GitHub Actions (cron: domingo 00:00).

También puede activarse manualmente desde la interfaz de GitHub Actions.

### Monitoreo

- **Health check**: Endpoint `/health` verifica disponibilidad del modelo
- **Logs**: Todos los eventos quedan registrados en stdout/stderr
- **Docker**: Healthcheck integrado en el contenedor

### Actualización del dataset

Para usar un dataset actualizado, modificar el dataset ID en `app/model/trainer.py`:

```python
dataset_path = kagglehub.dataset_download("arajmishra/potato-dataset")
```

---

## 👥 Equipo

**Proyecto Universitario** - Mayo 2026

*Equipo de hasta 4 estudiantes*

---

## 📝 Licencia

Este proyecto es creado con fines educativos.
