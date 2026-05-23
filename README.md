# 🛡️ StyloGuard: Advanced Indonesian Stylometry & Ghostwriter Detection

StyloGuard is a state-of-the-art system designed to detect human imposters and automated ghostwriters in Indonesian digital media. By leveraging a **Feature-Fusion Transformer** architecture, it combines the semantic power of pre-trained language models with the invariant nature of topic-blind stylometric features.

## 🚀 Overview & Key Features

StyloGuard addresses the growing challenge of digital authenticity. Traditional NLP models often struggle with "topic-leakage" where they identify the *subject* rather than the *author*. StyloGuard solves this by fusing semantic deep learning with topic-blind style fingerprints:

1.  **IndoBERT Contextual Backbone**: An Indonesian BERT architecture capturing deep contextual, semantic, and textual cues.
2.  **Topic-Blind Stylometrics**: 52 hand-crafted features (punctuation frequency, lexical diversity, structural patterns, part-of-speech distributions) capturing writing style invariant of topic.
3.  **💎 Premium Dual-Channel Explainable AI (xAI) Center**:
    *   **Semantic Channel (Inline Attention Heatmap)**: Highlights input text dynamically based on the exact self-attention weights extracted from the last layer of IndoBERT, allowing stakeholders to visually inspect word-level contributions.
    *   **Stylistic Channel (Autograd Driver Chart)**: Computes true stylometric feature contributions using PyTorch backpropagation (`Gradient * Input` attribution), displaying them in an elegant green/red positive/negative driver chart.

---

## 🏗️ Architecture & Optimizations

*   **Backend**: FastAPI (Python 3.12) exposing inference pipelines and database operations in under 100ms.
*   **Frontend**: React.js (Vite + TypeScript) with modern glassmorphic designs and smooth transitions.
*   **Deep Learning Engine**: PyTorch-based hybrid `FeatureFusionTransformer`.
*   **Optimized Docker Stack**: Engineered with **CPU-only PyTorch >=2.6** (vulnerability safe for CVE-2025-32434) and a **host-caching volume mount** (`~/.cache/huggingface`) that ensures near-instantaneous container starts.

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
│   ├── data/
│   │   ├── processed/
│   │   └── raw/
│   ├── scripts/
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── uv.lock
├── frontend/               # Vite React.js App
│   ├── src/                # Main Application Source
│   ├── public/             # Static Assets
│   └── package.json        # Dependencies
└── docker-compose.yml      # Orchestration
```

## 🛠️ Getting Started

### Prerequisites
- Python 3.12-3.13
- uv
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
    uv sync
    uv run python -m scripts.seed_db
    uv run uvicorn app.main:app --reload
    ```

3.  **Frontend Setup**
    ```bash
    cd ../frontend
    npm install
    npm run dev
    ```

4.  **Run with Docker (Recommended)**
    ```bash
    docker-compose up --build
    ```
    *   **Auto-Seeding**: Boots the SQLite database and seeds it with **3,904 historical articles** automatically.
    *   **Caching**: Maps the host's HuggingFace cache directory (`~/.cache/huggingface`) to the container for near-instant startups.
    *   **Ports**: Access the Web UI at `http://localhost:5173` and the backend API at `http://localhost:8000`.

---

## 🤖 Core Model Integration

The hybrid `FeatureFusionTransformer` resides inside `backend/app/model/`. At startup, the `ModelManager` singleton dynamically maps weights, tokenizer configurations, scaler instances, and labels directly from the `model_artifacts/` directory, gracefully degrading to a robust fallback state should any component be missing.
