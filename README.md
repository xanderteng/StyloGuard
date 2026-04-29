# 🛡️ StyloGuard: Advanced Indonesian Stylometry & Ghostwriter Detection

StyloGuard is a state-of-the-art system designed to detect human imposters and automated ghostwriters in Indonesian digital media. By leveraging a **Feature-Fusion Transformer** architecture, it combines the semantic power of pre-trained language models with the invariant nature of topic-blind stylometric features.

## 🚀 Overview

StyloGuard addresses the growing challenge of digital authenticity. Traditional NLP models often struggle with "topic-leakage" where they identify the *subject* rather than the *author*. StyloGuard solves this by fusing:
1.  **IndoBERT**: A state-of-the-art Indonesian BERT model for deep semantic and contextual understanding.
2.  **Topic-Blind Stylometrics**: Custom-extracted features (punctuation density, lexical diversity, syntactic patterns) that remain consistent regardless of the topic.

The result is a robust detector capable of identifying if a piece of text was written by the claimed author, a human imposter, or an LLM-generated ghostwriter.

## 🏗️ Architecture

-   **Backend**: FastAPI (Python) serving the Feature-Fusion Transformer model.
-   **Frontend**: React Native (Expo) with NativeWind for a premium, responsive cross-platform experience.
-   **Core Model**: PyTorch-based Transformer with fusion layers for heterogeneous feature integration.
-   **Containerization**: Fully dockerized environment for seamless deployment.

## 📂 Project Structure

```text
StyloGuard/
├── backend/                # FastAPI Application
│   ├── app/
│   │   ├── core/           # Config and Security
│   │   ├── db/             # Database Models and Session
│   │   ├── model/          # FF-Transformer & Stylometric Extractor
│   │   ├── routers/        # API Endpoints (Predict, Articles)
│   │   └── schemas/        # Pydantic Schemas
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/               # Expo React Native App
│   ├── app/                # Main Application Logic
│   ├── components/         # Reusable UI Components
│   └── tailwind.config.js  # NativeWind Styling
└── docker-compose.yml      # Orchestration
```

## 🛠️ Getting Started

### Prerequisites
- Python 3.10+
- Node.js & npm
- Docker & Docker Compose

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/StyloGuard.git
    cd StyloGuard
    ```

2.  **Backend Setup**
    ```bash
    cd backend
    python -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Frontend Setup**
    ```bash
    cd ../frontend
    npm install
    npx expo start
    ```

4.  **Run with Docker**
    ```bash
    docker-compose up --build
    ```

## 🤖 Core Model integration
The AI models for stylometric feature extraction and the pre-trained IndoBERT fusion are embedded directly into the `backend/app/model/` directory, ready to process requests efficiently via the FastAPI endpoints.
