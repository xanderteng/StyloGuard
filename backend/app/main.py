import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.session import Base, engine
from app.model.model_manager import ModelManager
from app.routers import articles, predict

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    Base.metadata.create_all(bind=engine)

    # Load PyTorch model, tokenizer, and scaler
    manager = ModelManager.get()
    manager.load()
    if manager.is_ready:
        logger.info("StyloGuard model loaded successfully.")
    else:
        logger.warning(
            "StyloGuard model NOT loaded — /predict will return 503. "
            "Place artefacts in backend/model_artifacts/ and restart."
        )

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("StyloGuard shutting down.")


app = FastAPI(
    title="StyloGuard API",
    description="Explainable Indonesian stylometry for authorship verification using Feature-Fusion Transformers.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(articles.router)
app.include_router(predict.router)


@app.get("/health")
def health_check() -> dict[str, str]:
    manager = ModelManager.get()
    return {
        "status": "ok",
        "model_loaded": str(manager.is_ready),
    }
