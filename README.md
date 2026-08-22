# AffectSense: Multimodal Emotion Recognition

FastAPI application that predicts emotion from text, a face image, and a WAV audio sample. The saved Keras model is loaded from `models/fusion_model.keras` when the application starts.

## Prerequisites

- The model file must exist at `models/fusion_model.keras`.
- For Docker: Docker Desktop or Docker Engine with the Compose plugin.
- For local Python: Python 3.10 and `pip`.

## Run with Docker Compose

From the project root, build the image and start the application:

```bash
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000).

To run it in the background:

```bash
docker compose up --build -d
```

View service logs or stop the service:

```bash
docker compose logs -f
docker compose down
```

## Run with Docker only

Build the image:

```bash
docker build -t affectsense .
```

Run the container:

```bash
docker run --rm -p 8000:8000 affectsense
```

## Run locally with a Python virtual environment

Create the virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies and start the development server:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Visit [http://localhost:8000](http://localhost:8000). Stop the server with `Ctrl+C`; use `deactivate` to leave the virtual environment.

## Inputs expected by the model

| Input | Upload requirement | Backend tensor shape |
| --- | --- | --- |
| Text | A non-empty short statement | `(1, 30)` |
| Face image | Any browser-supported image | `(1, 48, 48, 1)` grayscale, normalized |
| Voice sample | `.wav` only | `(1, 130, 40)` MFCC features |

The prediction call follows the model input order: face image, voice sample, then text.

## Project files

```text
app/main.py               FastAPI routes and model preprocessing
app/templates/index.html  Web interface
models/fusion_model.keras Keras model required at runtime
Dockerfile                Container image definition
docker-compose.yml        Compose service definition
```
