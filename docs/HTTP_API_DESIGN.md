# WanGP HTTP API Design Document

## Executive Summary

This document describes how to add a persistent HTTP API to WanGP that allows:
1. Loading models once at startup (or on-demand with caching)
2. Processing individual rendering requests quickly with pre-loaded models
3. Supporting both synchronous and asynchronous request handling

---

## Part 1: Current Architecture Analysis

### 1.1 Entry Points

The application (`wgp.py`) has two entry modes:

1. **Gradio Web UI Mode** (default): Launches a web interface on port 7860
2. **CLI Queue Processing Mode** (`--process`): Processes a queue file headlessly

```python
# wgp.py:10917-11035
if __name__ == "__main__":
    if len(args.process) > 0:
        # CLI mode - process queue.zip or settings.json
        process_tasks_cli(queue, state)
    else:
        # Gradio mode
        demo = create_ui()
        demo.launch(server_name=server_name, server_port=server_port, ...)
```

### 1.2 Global Model State

The application maintains global state for loaded models:

```python
# wgp.py:107-108
wan_model = None          # Current loaded model instance
offloadobj = None         # Memory/offload manager (mmgp library)
reload_needed = True      # Flag to force model reload
transformer_type = "t2v"  # Currently loaded model type identifier
loaded_profile = 0        # Current memory profile (0-5)
```

### 1.3 Model Loading Pipeline

Models are loaded via `load_models()` (wgp.py:3247-3362):

```python
def load_models(model_type, override_profile=-1, **model_kwargs):
    # 1. Get model definition and base type
    model_def = get_model_def(model_type)
    base_model_type = get_base_model_type(model_type)

    # 2. Download models if needed (HuggingFace)
    download_models(filename, model_type, ...)

    # 3. Load model via handler
    model_type_handler = model_types_handlers[base_model_type]
    wan_model, pipe = model_type_handler.load_model(...)

    # 4. Initialize memory management profile
    offloadobj = offload.profile(pipe, profile_no=mmgp_profile, ...)

    return wan_model, offloadobj
```

**Model handlers** are registered in `model_types_handlers` dict, mapping model types to handler modules:
- `models.wan.wan_handler` - Wan 2.1/2.2 models
- `models.hyvideo.hunyuan_handler` - HunyuanVideo
- `models.flux.flux_handler` - Flux models
- `models.ltx2.ltx2_handler` - LTX2 models
- etc.

Each handler implements:
- `query_supported_types()` - List of supported model type strings
- `load_model()` - Load checkpoint and return model + pipe dict
- `query_model_def()` - Model-specific configuration

### 1.4 Generation Pipeline

The core generation function is `generate_video()` (wgp.py:5217-6700):

```python
def generate_video(
    task,              # Task dict with id, prompt, params
    send_cmd,          # Callback for progress/status updates
    image_mode,        # 0=video, 1=image, 2=inpaint
    prompt,
    negative_prompt,
    resolution,
    video_length,
    # ... 100+ parameters
    state,
    model_type,
    mode,
    plugin_data=None,
):
    # 1. Check if model reload needed
    if model_type != transformer_type or reload_needed:
        wan_model, offloadobj = load_models(model_type, override_profile)

    # 2. Preprocess inputs (images, videos, audio, masks)
    # 3. Prepare conditioning (text encoding, image encoding)
    # 4. Run generation loop (with sliding window for long videos)
    samples = wan_model.generate(
        input_prompt=prompt,
        image_start=image_start_tensor,
        image_end=image_end_tensor,
        # ... many parameters
        callback=callback,
        offloadobj=offloadobj,
    )

    # 5. Postprocess (VAE decode, audio generation, upsampling)
    # 6. Save output files
```

### 1.5 Task Queue Structure

Tasks are stored as dictionaries:

```python
task = {
    "id": "unique_task_id_timestamp",
    "prompt": "A dragon flying over mountains",
    "params": {
        "model_type": "t2v",
        "resolution": "1024x576",
        "video_length": 145,
        "num_inference_steps": 30,
        "seed": 42,
        "guidance_scale": 7.5,
        "activated_loras": [],
        "loras_multipliers": "1.0",
        # ... 80+ more parameters
    },
    "start_image_data_base64": [...],  # Optional: base64 images
    "end_image_data_base64": [...],
    "attachments": {...}  # Media file references
}
```

### 1.6 Request Validation

`validate_settings()` (wgp.py:499) validates all inputs before generation:
- Prompt parsing and template processing
- Resolution validation against model capabilities
- Image/video attachment validation
- LoRA existence checking
- Model compatibility verification

### 1.7 Progress Communication

The `send_cmd()` callback system handles real-time updates:

```python
def send_cmd(cmd_type, data):
    # cmd_type: "progress", "status", "output", "error", "info", "exit"
    # Used by both Gradio UI and CLI mode
```

---

## Part 2: Proposed HTTP API Design

### 2.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        WanGP HTTP Server                        │
├─────────────────────────────────────────────────────────────────┤
│  FastAPI Application                                            │
│  ├── /models              - Model management endpoints          │
│  ├── /generate            - Synchronous generation              │
│  ├── /generate/async      - Async generation (returns job ID)   │
│  ├── /jobs/{id}           - Job status/results                  │
│  └── /health              - Health check                        │
├─────────────────────────────────────────────────────────────────┤
│  Model Manager (Singleton)                                      │
│  ├── Loaded models cache                                        │
│  ├── Model preloading on startup                                │
│  └── Automatic model switching                                  │
├─────────────────────────────────────────────────────────────────┤
│  Job Queue (Background Tasks)                                   │
│  ├── Async job submission                                       │
│  ├── Progress tracking                                          │
│  └── Result storage                                             │
├─────────────────────────────────────────────────────────────────┤
│  Existing Generation Pipeline (wgp.py)                          │
│  ├── generate_video()                                           │
│  ├── Model handlers                                             │
│  └── Preprocessing/Postprocessing                               │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 API Endpoints

#### Health & Info

```
GET /health
Response: {"status": "ok", "version": "10.43", "loaded_model": "t2v" | null}

GET /models
Response: {
    "available": ["t2v", "i2v", "vace_14B", ...],
    "loaded": "t2v" | null,
    "families": {"wan": [...], "hunyuan": [...], ...}
}

GET /models/{model_type}
Response: {
    "model_type": "t2v",
    "family": "wan",
    "supported_modes": ["text2video", "video2video"],
    "default_settings": {...}
}
```

#### Model Management

```
POST /models/load
Body: {"model_type": "t2v", "profile": 3}
Response: {"status": "loading" | "loaded", "model_type": "t2v"}

POST /models/unload
Response: {"status": "unloaded"}
```

#### Synchronous Generation

```
POST /generate
Content-Type: application/json (or multipart/form-data for files)

Body: {
    "prompt": "A dragon flying over mountains",
    "model_type": "t2v",            // Optional if model pre-loaded
    "resolution": "1024x576",
    "video_length": 145,
    "num_inference_steps": 30,
    "seed": 42,                      // Optional, random if omitted
    "guidance_scale": 7.5,
    "negative_prompt": "",

    // Optional image inputs (base64 or URLs)
    "image_start": "data:image/png;base64,...",
    "image_end": null,

    // Optional advanced settings
    "loras": [{"name": "style.safetensors", "weight": 1.0}],
    "flow_shift": 3.0,
    "sample_solver": "unipc",

    // Output options
    "output_format": "mp4",          // mp4, gif, frames
    "return_base64": false           // Return file path or base64
}

Response (success): {
    "status": "completed",
    "output_path": "/outputs/video_001.mp4",
    "output_base64": null,           // If return_base64=true
    "metadata": {
        "seed": 42,
        "generation_time": 45.2,
        "model_type": "t2v"
    }
}

Response (error): {
    "status": "error",
    "error": "Insufficient VRAM for requested resolution",
    "error_type": "vram_error"
}
```

#### Asynchronous Generation

```
POST /generate/async
Body: (same as /generate)

Response: {
    "job_id": "job_abc123",
    "status": "queued"
}

GET /jobs/{job_id}
Response: {
    "job_id": "job_abc123",
    "status": "processing" | "completed" | "failed" | "queued",
    "progress": {
        "step": 15,
        "total_steps": 30,
        "phase": "Denoising",
        "eta_seconds": 20
    },
    "result": null | {
        "output_path": "/outputs/video_001.mp4",
        "metadata": {...}
    },
    "error": null | "Error message"
}

DELETE /jobs/{job_id}
Response: {"status": "cancelled"}

GET /jobs
Query: ?status=processing&limit=10
Response: {
    "jobs": [{...}, {...}],
    "total": 25
}
```

#### Streaming Progress (WebSocket)

```
WS /ws/jobs/{job_id}
Messages (server -> client):
{
    "type": "progress",
    "step": 15,
    "total_steps": 30,
    "phase": "Denoising"
}
{
    "type": "preview",
    "frame_base64": "..."    // Optional live preview
}
{
    "type": "completed",
    "output_path": "..."
}
{
    "type": "error",
    "message": "..."
}
```

### 2.3 Request/Response Models (Pydantic)

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from enum import Enum

class LoraConfig(BaseModel):
    name: str
    weight: float = 1.0

class GenerationRequest(BaseModel):
    # Required
    prompt: str

    # Model selection (optional if pre-loaded)
    model_type: Optional[str] = None

    # Core generation parameters
    resolution: str = "1024x576"
    video_length: int = 145
    num_inference_steps: int = 30
    seed: Optional[int] = None
    guidance_scale: float = 7.5
    negative_prompt: str = ""

    # Image inputs (base64 or file paths)
    image_start: Optional[str] = None
    image_end: Optional[str] = None
    image_refs: Optional[List[str]] = None

    # Video inputs
    video_source: Optional[str] = None
    video_guide: Optional[str] = None

    # Advanced generation settings
    loras: List[LoraConfig] = []
    flow_shift: float = 3.0
    sample_solver: str = "unipc"
    embedded_guidance_scale: float = 0.0

    # Sliding window (for long videos)
    sliding_window_size: int = 0
    sliding_window_overlap: int = 0

    # Output configuration
    output_format: Literal["mp4", "gif", "frames"] = "mp4"
    return_base64: bool = False
    output_filename: Optional[str] = None

class GenerationResponse(BaseModel):
    status: Literal["completed", "error"]
    output_path: Optional[str] = None
    output_base64: Optional[str] = None
    metadata: dict = {}
    error: Optional[str] = None
    error_type: Optional[str] = None

class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "completed", "failed", "cancelled"]
    progress: Optional[dict] = None
    result: Optional[GenerationResponse] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str
```

---

## Part 3: Implementation Plan

### 3.1 New File Structure

```
Wan2GP/
├── wgp.py                    # Existing (keep unchanged)
├── api/
│   ├── __init__.py
│   ├── server.py             # FastAPI application
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── models.py         # Model management endpoints
│   │   ├── generate.py       # Generation endpoints
│   │   └── jobs.py           # Job management endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── model_manager.py  # Singleton model manager
│   │   ├── generator.py      # Generation service wrapper
│   │   └── job_queue.py      # Async job queue
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── requests.py       # Pydantic request models
│   │   └── responses.py      # Pydantic response models
│   └── utils/
│       ├── __init__.py
│       └── media.py          # Base64/file handling
├── wgp_api.py                # New entry point for API server
└── requirements-api.txt      # Additional dependencies
```

### 3.2 Core Components

#### 3.2.1 Model Manager (Singleton)

```python
# api/services/model_manager.py
import threading
from typing import Optional, Dict, Any

class ModelManager:
    """Singleton that manages model loading and caching."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.wan_model = None
        self.offloadobj = None
        self.current_model_type: Optional[str] = None
        self.model_lock = threading.Lock()

    def load_model(self, model_type: str, profile: int = -1) -> bool:
        """Load a model, replacing any currently loaded model."""
        with self.model_lock:
            if self.current_model_type == model_type:
                return True  # Already loaded

            # Import from wgp.py
            from wgp import load_models, release_model

            # Unload current model
            if self.wan_model is not None:
                release_model()

            # Load new model
            self.wan_model, self.offloadobj = load_models(
                model_type,
                override_profile=profile
            )
            self.current_model_type = model_type
            return True

    def unload_model(self):
        """Unload the current model to free memory."""
        with self.model_lock:
            from wgp import release_model
            release_model()
            self.wan_model = None
            self.offloadobj = None
            self.current_model_type = None

    def get_model(self):
        """Get the currently loaded model and offload object."""
        return self.wan_model, self.offloadobj

    def is_loaded(self, model_type: str = None) -> bool:
        """Check if a model is loaded (optionally check specific type)."""
        if model_type:
            return self.current_model_type == model_type
        return self.wan_model is not None

# Global instance
model_manager = ModelManager()
```

#### 3.2.2 Generation Service

```python
# api/services/generator.py
import inspect
import uuid
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass

@dataclass
class GenerationResult:
    success: bool
    output_path: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = None

class GeneratorService:
    """Wraps the existing generate_video function for API use."""

    def __init__(self, model_manager):
        self.model_manager = model_manager

    def generate(
        self,
        params: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> GenerationResult:
        """
        Execute a generation request.

        Args:
            params: Generation parameters (prompt, resolution, etc.)
            progress_callback: Optional callback for progress updates
        """
        from wgp import (
            generate_video,
            validate_settings,
            primary_settings,
            get_gen_info
        )

        # Ensure model is loaded
        model_type = params.get("model_type")
        if model_type and not self.model_manager.is_loaded(model_type):
            self.model_manager.load_model(model_type)

        # Create task structure
        task_id = f"api_{uuid.uuid4().hex[:8]}"
        task = {
            "id": task_id,
            "prompt": params.get("prompt", ""),
            "params": params
        }

        # Create state
        state = self._create_state()

        # Validate settings
        inputs = primary_settings.copy()
        inputs.update(params)
        inputs["prompt"] = task["prompt"]
        inputs.setdefault("mode", "")

        override_inputs, _, _, _ = validate_settings(
            state,
            model_type,
            single_prompt=True,
            inputs=inputs
        )

        if override_inputs is None:
            return GenerationResult(
                success=False,
                error="Settings validation failed",
                error_type="validation_error"
            )

        inputs.update(override_inputs)
        inputs["state"] = state

        # Setup progress callback
        result_container = {"output_path": None, "error": None}

        def send_cmd(cmd_type, data=None):
            if cmd_type == "output":
                # Capture output path from state
                gen = get_gen_info(state)
                if gen.get("file_list"):
                    result_container["output_path"] = gen["file_list"][-1]
            elif cmd_type == "error":
                result_container["error"] = str(data)
            elif cmd_type == "progress" and progress_callback:
                progress_callback(cmd_type, data)
            elif progress_callback:
                progress_callback(cmd_type, data)

        # Filter to valid generate_video parameters
        expected_args = set(inspect.signature(generate_video).parameters.keys())
        filtered_params = {k: v for k, v in inputs.items() if k in expected_args}

        try:
            generate_video(task, send_cmd, **filtered_params)

            if result_container["error"]:
                return GenerationResult(
                    success=False,
                    error=result_container["error"],
                    error_type="generation_error"
                )

            return GenerationResult(
                success=True,
                output_path=result_container["output_path"],
                metadata={
                    "seed": params.get("seed"),
                    "model_type": model_type
                }
            )

        except Exception as e:
            return GenerationResult(
                success=False,
                error=str(e),
                error_type="exception"
            )

    def _create_state(self) -> Dict:
        """Create a minimal state dict for generation."""
        return {
            "gen": {
                "queue": [],
                "in_progress": False,
                "file_list": [],
                "file_settings_list": [],
                "audio_file_list": [],
                "audio_file_settings_list": [],
                "selected": 0,
                "audio_selected": 0,
                "prompt_no": 1,
                "prompts_max": 1,
                "repeat_no": 1,
                "total_generation": 1,
                "window_no": 1,
                "total_windows": 1,
                "progress_status": "",
                "process_status": "process:main",
            },
            "loras": [],
        }
```

#### 3.2.3 Job Queue

```python
# api/services/job_queue.py
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading

class JobStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Job:
    id: str
    params: Dict
    status: JobStatus = JobStatus.QUEUED
    progress: Optional[Dict] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self):
        return {
            "job_id": self.id,
            "status": self.status.value,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class JobQueue:
    """Manages async generation jobs."""

    def __init__(self, generator_service, max_workers: int = 1):
        self.generator = generator_service
        self.jobs: Dict[str, Job] = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.progress_callbacks: Dict[str, list] = {}

    def submit(self, params: Dict) -> str:
        """Submit a new generation job."""
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = Job(id=job_id, params=params)

        with self.lock:
            self.jobs[job_id] = job

        # Submit to thread pool
        self.executor.submit(self._process_job, job_id)

        return job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job."""
        with self.lock:
            job = self.jobs.get(job_id)
            if job and job.status == JobStatus.QUEUED:
                job.status = JobStatus.CANCELLED
                job.updated_at = datetime.utcnow()
                return True
        return False

    def register_progress_callback(self, job_id: str, callback: Callable):
        """Register a callback for progress updates."""
        if job_id not in self.progress_callbacks:
            self.progress_callbacks[job_id] = []
        self.progress_callbacks[job_id].append(callback)

    def _process_job(self, job_id: str):
        """Process a job (runs in thread pool)."""
        job = self.jobs.get(job_id)
        if not job or job.status == JobStatus.CANCELLED:
            return

        # Update status
        job.status = JobStatus.PROCESSING
        job.updated_at = datetime.utcnow()

        def progress_callback(cmd_type, data):
            job.updated_at = datetime.utcnow()
            if cmd_type == "progress":
                if isinstance(data, list) and len(data) >= 2:
                    if isinstance(data[0], tuple):
                        step, total = data[0]
                    else:
                        step, total = 0, 1
                    job.progress = {
                        "step": step,
                        "total_steps": total,
                        "phase": data[1] if len(data) > 1 else ""
                    }

            # Notify registered callbacks
            for cb in self.progress_callbacks.get(job_id, []):
                try:
                    cb(cmd_type, data)
                except:
                    pass

        # Run generation
        result = self.generator.generate(job.params, progress_callback)

        # Update job with result
        job.updated_at = datetime.utcnow()
        if result.success:
            job.status = JobStatus.COMPLETED
            job.result = {
                "output_path": result.output_path,
                "metadata": result.metadata
            }
        else:
            job.status = JobStatus.FAILED
            job.error = result.error

        # Cleanup callbacks
        self.progress_callbacks.pop(job_id, None)
```

#### 3.2.4 FastAPI Application

```python
# api/server.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os
import sys

# Add parent directory to path for wgp imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.services.model_manager import model_manager
from api.services.generator import GeneratorService
from api.services.job_queue import JobQueue
from api.schemas.requests import GenerationRequest, LoadModelRequest
from api.schemas.responses import (
    GenerationResponse,
    JobStatusResponse,
    ModelInfoResponse,
    HealthResponse
)

# Services (initialized on startup)
generator_service = None
job_queue = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global generator_service, job_queue

    generator_service = GeneratorService(model_manager)
    job_queue = JobQueue(generator_service, max_workers=1)

    # Optional: preload default model
    preload_model = os.getenv("WANGP_PRELOAD_MODEL")
    if preload_model:
        print(f"Preloading model: {preload_model}")
        model_manager.load_model(preload_model)

    yield

    # Cleanup
    model_manager.unload_model()

app = FastAPI(
    title="WanGP API",
    description="HTTP API for WanGP video generation",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health endpoint
@app.get("/health", response_model=HealthResponse)
async def health():
    return {
        "status": "ok",
        "version": "10.43",
        "loaded_model": model_manager.current_model_type
    }

# Model endpoints
@app.get("/models")
async def list_models():
    from wgp import model_types_handlers

    available = []
    for handler in model_types_handlers.values():
        available.extend(handler.query_supported_types())

    return {
        "available": sorted(set(available)),
        "loaded": model_manager.current_model_type
    }

@app.post("/models/load")
async def load_model(request: LoadModelRequest):
    try:
        model_manager.load_model(request.model_type, request.profile)
        return {
            "status": "loaded",
            "model_type": request.model_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/unload")
async def unload_model():
    model_manager.unload_model()
    return {"status": "unloaded"}

# Synchronous generation
@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    params = request.model_dump(exclude_none=True)
    result = generator_service.generate(params)

    if not result.success:
        return GenerationResponse(
            status="error",
            error=result.error,
            error_type=result.error_type
        )

    return GenerationResponse(
        status="completed",
        output_path=result.output_path,
        metadata=result.metadata or {}
    )

# Async generation
@app.post("/generate/async")
async def generate_async(request: GenerationRequest):
    params = request.model_dump(exclude_none=True)
    job_id = job_queue.submit(params)
    return {"job_id": job_id, "status": "queued"}

@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()

@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    if job_queue.cancel_job(job_id):
        return {"status": "cancelled"}
    raise HTTPException(status_code=400, detail="Cannot cancel job")

# File download
@app.get("/files/{filename:path}")
async def download_file(filename: str):
    from wgp import save_path
    file_path = os.path.join(save_path, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)
```

### 3.3 Entry Point Script

```python
# wgp_api.py
#!/usr/bin/env python3
"""
WanGP HTTP API Server

Usage:
    python wgp_api.py [options]

Options:
    --host HOST         Server host (default: 0.0.0.0)
    --port PORT         Server port (default: 8000)
    --preload MODEL     Preload model on startup
    --profile PROFILE   Memory profile for preloaded model (0-5)
    --workers N         Number of generation workers (default: 1)
"""

import os
import sys
import argparse

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="WanGP HTTP API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--preload", type=str, help="Model to preload on startup")
    parser.add_argument("--profile", type=int, default=-1, help="Memory profile (0-5)")
    parser.add_argument("--workers", type=int, default=1, help="Generation workers")
    args = parser.parse_args()

    # Set environment variables for API server
    if args.preload:
        os.environ["WANGP_PRELOAD_MODEL"] = args.preload
    os.environ["WANGP_PROFILE"] = str(args.profile)
    os.environ["WANGP_WORKERS"] = str(args.workers)

    import uvicorn
    from api.server import app

    print(f"Starting WanGP API server on {args.host}:{args.port}")
    if args.preload:
        print(f"Will preload model: {args.preload}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )

if __name__ == "__main__":
    main()
```

### 3.4 Dependencies

```
# requirements-api.txt
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
python-multipart>=0.0.6
websockets>=12.0
```

---

## Part 4: Integration Strategy

### 4.1 Modifications to wgp.py

The existing `wgp.py` requires minimal modifications:

1. **Make functions importable**: Ensure key functions don't rely on Gradio-specific context
2. **Expose model state**: The global variables are already accessible
3. **Add API-friendly validation**: Modify `validate_settings()` to work without Gradio state

```python
# Potential minimal changes to wgp.py

# 1. Make release_model() idempotent (already is)
# 2. Ensure generate_video() can work with minimal state
# 3. Add a clean interface for the API to use

def get_model_types_handlers():
    """Public accessor for model handlers."""
    return model_types_handlers

def get_current_model_info():
    """Get info about currently loaded model."""
    return {
        "model_type": transformer_type,
        "loaded": wan_model is not None,
        "profile": loaded_profile
    }
```

### 4.2 Running Both Gradio and HTTP API

Option 1: **Separate processes** (recommended for production)
```bash
# Terminal 1: Gradio UI
python wgp.py

# Terminal 2: HTTP API
python wgp_api.py --preload t2v
```

Option 2: **Combined process** with shared model state
```python
# Add to wgp.py main block
if args.api_mode:
    # Start both Gradio and FastAPI
    import threading
    import uvicorn
    from api.server import app

    api_thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": "0.0.0.0", "port": 8000},
        daemon=True
    )
    api_thread.start()

    # Continue with Gradio
    demo.launch(...)
```

---

## Part 5: Usage Examples

### 5.1 Simple Text-to-Video

```python
import requests

# Load model
requests.post("http://localhost:8000/models/load", json={
    "model_type": "t2v",
    "profile": 3
})

# Generate video
response = requests.post("http://localhost:8000/generate", json={
    "prompt": "A majestic dragon flying through clouds at sunset",
    "resolution": "1024x576",
    "video_length": 97,
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "seed": 42
})

result = response.json()
print(f"Video saved to: {result['output_path']}")
```

### 5.2 Image-to-Video

```python
import requests
import base64

# Read and encode image
with open("start_image.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Generate with start image
response = requests.post("http://localhost:8000/generate", json={
    "prompt": "The person starts walking forward",
    "model_type": "i2v",
    "image_start": f"data:image/png;base64,{image_b64}",
    "resolution": "1024x576",
    "video_length": 97,
    "num_inference_steps": 30
})
```

### 5.3 Async Generation with Progress

```python
import requests
import time

# Submit async job
response = requests.post("http://localhost:8000/generate/async", json={
    "prompt": "A beautiful waterfall in a forest",
    "model_type": "t2v",
    "video_length": 145
})
job_id = response.json()["job_id"]

# Poll for progress
while True:
    status = requests.get(f"http://localhost:8000/jobs/{job_id}").json()

    if status["status"] == "processing":
        progress = status.get("progress", {})
        print(f"Step {progress.get('step', '?')}/{progress.get('total_steps', '?')}")
    elif status["status"] == "completed":
        print(f"Done! Output: {status['result']['output_path']}")
        break
    elif status["status"] == "failed":
        print(f"Failed: {status['error']}")
        break

    time.sleep(1)
```

### 5.4 Python Client Library

```python
# Example client wrapper
class WanGPClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def load_model(self, model_type, profile=-1):
        resp = requests.post(f"{self.base_url}/models/load", json={
            "model_type": model_type,
            "profile": profile
        })
        resp.raise_for_status()
        return resp.json()

    def generate(self, prompt, **kwargs):
        resp = requests.post(f"{self.base_url}/generate", json={
            "prompt": prompt,
            **kwargs
        })
        resp.raise_for_status()
        return resp.json()

    def generate_async(self, prompt, **kwargs):
        resp = requests.post(f"{self.base_url}/generate/async", json={
            "prompt": prompt,
            **kwargs
        })
        resp.raise_for_status()
        return resp.json()["job_id"]

    def wait_for_job(self, job_id, poll_interval=1.0, callback=None):
        while True:
            resp = requests.get(f"{self.base_url}/jobs/{job_id}")
            resp.raise_for_status()
            status = resp.json()

            if callback:
                callback(status)

            if status["status"] in ("completed", "failed", "cancelled"):
                return status

            time.sleep(poll_interval)

# Usage
client = WanGPClient()
client.load_model("t2v")
result = client.generate("A cat playing piano", video_length=97)
print(result)
```

---

## Part 6: Considerations

### 6.1 Thread Safety

- Model loading/unloading must be serialized (only one model at a time)
- Generation can only run one at a time (GPU constraint)
- Job queue handles serialization automatically

### 6.2 Memory Management

- Models consume 10-40GB+ of VRAM/RAM
- Only one model can be loaded at a time on typical hardware
- Consider implementing model unloading after idle timeout

### 6.3 Error Handling

- VRAM exhaustion: Return specific error type, suggest lower resolution
- Model not found: Return 404 with available models list
- Invalid parameters: Return 422 with validation details

### 6.4 Security Considerations

- Implement rate limiting for production
- Add authentication if exposed publicly
- Validate file paths to prevent directory traversal
- Sanitize output filenames

### 6.5 Future Enhancements

- WebSocket support for real-time progress streaming
- Batch generation endpoint
- Model warm-up endpoint (load and run small test)
- Metrics/monitoring endpoints
- Queue persistence across restarts
- Multi-GPU support with model sharding

---

## Summary

This design enables WanGP to function as a persistent HTTP service by:

1. **Wrapping existing code**: The `generate_video()` function and model handlers remain unchanged
2. **Adding a model manager**: Singleton that maintains loaded model state
3. **Providing REST endpoints**: FastAPI server with sync/async generation
4. **Supporting job queuing**: Background task processing with progress tracking

The implementation requires minimal changes to existing code while providing a clean API for external applications to use WanGP's video generation capabilities.
