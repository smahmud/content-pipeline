"""
Content Pipeline REST API

FastAPI application exposing pipeline operations via HTTP endpoints.

Usage:
    uvicorn api.app:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import APIConfig
from api.routers import health, extract, transcribe, enrich, format, validate, pipeline

config = APIConfig.load()

app = FastAPI(
    title="Content Pipeline API",
    description="REST API for extracting, transcribing, enriching, formatting, and validating multimedia content.",
    version="0.10.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers under /api/v1 prefix
PREFIX = "/api/v1"
app.include_router(health.router, prefix=PREFIX, tags=["Health"])
app.include_router(extract.router, prefix=PREFIX, tags=["Pipeline"])
app.include_router(transcribe.router, prefix=PREFIX, tags=["Pipeline"])
app.include_router(enrich.router, prefix=PREFIX, tags=["Pipeline"])
app.include_router(format.router, prefix=PREFIX, tags=["Pipeline"])
app.include_router(validate.router, prefix=PREFIX, tags=["Pipeline"])
app.include_router(pipeline.router, prefix=PREFIX, tags=["Pipeline"])


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Content Pipeline API",
        "version": "0.10.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
