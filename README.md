# Multimodal Emotion Recognition System

An advanced deep learning-based system designed to recognize and analyze human emotions by fusing multiple input modalities: **Text, Face Image, and Voice Audio**.

---

## 1. Project Overview & Objectives
In modern human-computer interaction, relying on a single mode of communication often leads to inaccurate emotional assessments. This project bridges that gap by implementing a **multimodal data fusion** architecture. 

**Core Objectives:**
* To integrate multiple data streams (textual, visual, and auditory) into a unified neural network model.
* To provide accurate emotional classification with transparent confidence scoring.
* To deliver a seamless user experience via a web interface backed by a high-performance FastAPI service.

---

## 2. Expected Inputs & Tensor Shapes
The model processes inputs sequentially according to specific structural requirements:

| Modality | Input Type | Description & Requirements | Backend Tensor Shape |
| :--- | :--- | :--- | :--- |
| **Face Image** | Image File | Any browser-supported image (Converted to grayscale & normalized) | `(1, 48, 48, 1)` |
| **Voice Sample** | `.wav` Audio | Audio file processed for speech features | `(1, 130, 40)` (MFCC features) |
| **Text** | String | A non-empty short textual statement | `(1, 30)` |

---

## 3. Project Structure
```text
multimodal-emotion-app/
│
├── app/
│   ├── main.py               # FastAPI routes and preprocessing logic
│   └── templates/
│       └── index.html        # Web interface for real-time interaction
│
├── images/
│   ├── evaluation.png        # Model evaluation graphs and metrics
│   └── api_outcome.png       # API prediction outcome screenshot
│
├── models/
│   └── fusion_model.keras    # Pre-trained Keras multimodal model
│
├── .dockerignore             # Files to ignore during Docker build
├── .gitignore                # Files to ignore in Git version control
├── Dockerfile                # Container image definition
├── README.md                 # Project documentation
├── docker-compose.yml        # Docker compose configuration
├── project NOTEBOOK...       # Jupyter notebook for model training/analysis
└── requirements.txt          # Python package dependencies

```

---

## 4. Prerequisites

* **Docker Desktop** installed and running on your system.
* Pre-trained model file correctly placed at `models/fusion_model.keras`.

---

## 5. Installation & Execution

To run the application smoothly using Docker, follow these steps:

1. **Clone the repository and navigate to the project directory:**
```bash
cd multimodal-emotion-app

```


2. **Build and start the application using Docker Compose:**
```bash
docker compose up --build

```


3. **Access the Application:**
Open your web browser and navigate to:
```text
http://localhost:8000

```


4. **Stop the Service:**
To stop the application container, run:
```bash
docker compose down

```



---

## 6. Expected Outcomes

* **Holistic Analysis:** Combines what a user types, how they look, and how they sound to predict the true underlying emotion.
* **Confidence Metrics:** Outputs predicted emotion categories along with percentage-based confidence scores.
* **Scalable Deployment:** Containerized via Docker to ensure consistency across different deployment environments.

```

```
