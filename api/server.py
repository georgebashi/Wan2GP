"""
WanGP HTTP API Server

FastAPI application providing REST endpoints for video generation.
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Ensure parent directory is in path for wgp imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.services.model_manager import model_manager
from api.services.generator import GeneratorService
from api.schemas.requests import GenerationRequest, LoadModelRequest
from api.schemas.responses import (
    GenerationResponse,
    ModelsResponse,
    LoadModelResponse,
    HealthResponse,
    ErrorResponse,
)

# Version (will be updated from wgp on startup)
VERSION = "1.0.0"

# Services (initialized on startup)
generator_service: Optional[GeneratorService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup, cleanup on shutdown."""
    global generator_service, VERSION

    # Initialize generator service
    generator_service = GeneratorService(model_manager)

    # Try to get version from wgp
    try:
        import wgp
        VERSION = getattr(wgp, "WanGP_version", "1.0.0")
    except:
        pass

    # Preload model if configured
    preload_model = os.getenv("WANGP_PRELOAD_MODEL")
    preload_profile = int(os.getenv("WANGP_PROFILE", "-1"))

    if preload_model:
        print(f"Preloading model: {preload_model}")
        try:
            model_manager.load_model(preload_model, profile=preload_profile)
            print(f"Model {preload_model} loaded successfully")
        except Exception as e:
            print(f"Failed to preload model: {e}")

    yield

    # Cleanup (optional: unload model to free memory)
    # model_manager.unload_model()


app = FastAPI(
    title="WanGP API",
    description="HTTP API for WanGP video generation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health & Info Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Check server health and get basic info."""
    return HealthResponse(
        status="ok",
        version=VERSION,
        loaded_model=model_manager.get_current_model_type()
    )


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "WanGP API",
        "version": VERSION,
        "docs": "/docs",
        "health": "/health"
    }


# =============================================================================
# Model Management Endpoints
# =============================================================================

@app.get("/models", response_model=ModelsResponse, tags=["Models"])
async def list_models():
    """List all available models and current loaded model."""
    try:
        families = model_manager.get_available_models()
        all_types = model_manager.get_all_model_types()

        return ModelsResponse(
            available=all_types,
            loaded=model_manager.get_current_model_type(),
            families=families
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/load", response_model=LoadModelResponse, tags=["Models"])
async def load_model(request: LoadModelRequest):
    """
    Load a model into memory.

    This will replace any currently loaded model. The model remains loaded
    for subsequent generation requests until a different model is loaded.
    """
    try:
        model_manager.load_model(request.model_type, profile=request.profile)
        return LoadModelResponse(
            status="loaded",
            model_type=request.model_type
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )


# =============================================================================
# Generation Endpoints
# =============================================================================

@app.post("/generate", response_model=GenerationResponse, tags=["Generation"])
async def generate(request: GenerationRequest):
    """
    Generate a video or image.

    If no model is currently loaded, you must specify model_type.
    If a model is already loaded, you can omit model_type to use it,
    or specify a different model_type to switch models.

    Returns the path to the generated file on success.
    """
    if generator_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Convert request to dict, excluding None values
    params = request.model_dump(exclude_none=True)

    # Run generation
    result = generator_service.generate(params)

    if not result.success:
        return GenerationResponse(
            status="error",
            error=result.error,
            error_type=result.error_type,
            metadata={}
        )

    return GenerationResponse(
        status="completed",
        output_path=result.output_path,
        metadata=result.metadata or {}
    )


# =============================================================================
# File Serving Endpoints
# =============================================================================

@app.get("/files/{filepath:path}", tags=["Files"])
async def get_file(filepath: str):
    """
    Download a generated file by path.

    The filepath should be relative to the outputs directory.
    """
    try:
        import wgp
        save_path = wgp.save_path
    except:
        save_path = "outputs"

    # Handle both absolute and relative paths
    if os.path.isabs(filepath):
        full_path = filepath
    else:
        full_path = os.path.join(save_path, filepath)

    # Security: ensure path is within allowed directory
    full_path = os.path.abspath(full_path)
    save_path_abs = os.path.abspath(save_path)

    if not full_path.startswith(save_path_abs) and not os.path.exists(full_path):
        raise HTTPException(status_code=403, detail="Access denied")

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(full_path)


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    return ErrorResponse(
        error=str(exc),
        error_type="internal_error",
        detail=None
    )
