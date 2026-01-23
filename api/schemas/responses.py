"""
Response schemas for the WanGP HTTP API.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Literal


class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["ok", "error"] = Field(..., description="Server status")
    version: str = Field(..., description="WanGP version")
    loaded_model: Optional[str] = Field(None, description="Currently loaded model type")


class ModelInfo(BaseModel):
    """Information about a model."""
    model_type: str = Field(..., description="Model type identifier")
    family: str = Field(..., description="Model family")
    loaded: bool = Field(..., description="Whether this model is currently loaded")


class ModelsResponse(BaseModel):
    """Response listing available models."""
    available: List[str] = Field(..., description="Available model types")
    loaded: Optional[str] = Field(None, description="Currently loaded model type")
    families: Dict[str, List[str]] = Field(..., description="Models grouped by family")


class LoadModelResponse(BaseModel):
    """Response after loading a model."""
    status: Literal["loaded", "error"] = Field(..., description="Load status")
    model_type: str = Field(..., description="Loaded model type")
    error: Optional[str] = Field(None, description="Error message if failed")


class GenerationMetadata(BaseModel):
    """Metadata about a generation."""
    seed: Optional[int] = Field(None, description="Seed used")
    model_type: Optional[str] = Field(None, description="Model type used")
    resolution: Optional[str] = Field(None, description="Output resolution")
    video_length: Optional[int] = Field(None, description="Number of frames")
    num_inference_steps: Optional[int] = Field(None, description="Denoising steps")


class GenerationResponse(BaseModel):
    """Response from a generation request."""
    status: Literal["completed", "error"] = Field(..., description="Generation status")
    output_path: Optional[str] = Field(None, description="Path to generated file")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
    error: Optional[str] = Field(None, description="Error message if failed")
    error_type: Optional[str] = Field(None, description="Error type if failed")


class ErrorResponse(BaseModel):
    """Generic error response."""
    error: str = Field(..., description="Error message")
    error_type: str = Field("unknown", description="Error type")
    detail: Optional[str] = Field(None, description="Additional details")
