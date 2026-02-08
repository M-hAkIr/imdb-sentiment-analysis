import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.config import (
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_ORIGINS,
    MAX_TEXT_LENGTH,
    MIN_TEXT_LENGTH,
    TEXT_TRUNCATE_LENGTH,
)
from src.models import SentimentAnalyzer, create_analyzer, normalize_label
from src.preprocessing import preprocess_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Request/Response schemas 

class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=MIN_TEXT_LENGTH, max_length=MAX_TEXT_LENGTH)
    model: Literal["distilbert", "roberta"] = Field(default="distilbert")

    @field_validator('text')
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()


class PredictionResponse(BaseModel):
    text: str
    sentiment: Literal["positive", "negative"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_used: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: Literal["healthy", "unhealthy"]
    models_loaded: list[str]


# App

models: dict[str, SentimentAnalyzer] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading sentiment models...")
    try:
        models["distilbert"] = create_analyzer("distilbert")
        models["roberta"] = create_analyzer("roberta")
        logger.info("Models ready: %s", list(models.keys()))
    except Exception as e:
        logger.error("Model loading failed: %s", e)
        raise RuntimeError("Cannot start without models") from e

    yield
    models.clear()


app = FastAPI(
    title="IMDB Sentiment Analysis API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _truncate(text: str, max_len: int = TEXT_TRUNCATE_LENGTH) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


# Endpoints

@app.get("/")
async def root():
    return {"message": "IMDB Sentiment Analysis API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    healthy = bool(models)
    response = HealthResponse(
        status="healthy" if healthy else "unhealthy",
        models_loaded=list(models.keys()),
    )
    if not healthy:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response.model_dump(),
        )
    return response


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict sentiment for a single text."""
    cleaned = preprocess_text(request.text)
    if not cleaned or len(cleaned.strip()) < MIN_TEXT_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Text too short after preprocessing (min {MIN_TEXT_LENGTH} chars)",
        )

    analyzer = models[request.model]
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: analyzer.predict(cleaned))
    sentiment = normalize_label(result.label)

    return PredictionResponse(
        text=_truncate(request.text),
        sentiment=sentiment.value,
        confidence=result.confidence,
        model_used=request.model,
        inference_time_ms=result.inference_time * 1000,
    )


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Validation Error", "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal Server Error", "detail": "Something went wrong"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
