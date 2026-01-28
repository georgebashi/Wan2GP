"""
Request schemas for the WanGP HTTP API.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ModelDefinition(BaseModel):
    """Dynamic model definition for runtime registration."""
    architecture: str = Field(..., description="Base model type (e.g., 'i2v_2_2', 't2v')")
    URLs: List[str] = Field(..., description="Primary model file paths")
    URLs2: Optional[List[str]] = Field(None, description="Secondary model files (e.g., for quality switching)")
    text_encoder_URLs: Optional[List[str]] = Field(None, description="Text encoder file paths")
    VAE_URLs: Optional[List[str]] = Field(None, description="VAE file paths")
    name: Optional[str] = Field(None, description="Human-readable model name")
    group: Optional[str] = Field(None, description="Model family group")

    class Config:
        extra = "allow"  # Allow additional fields for model-specific options


class LoraConfig(BaseModel):
    """Configuration for a LoRA adapter."""
    name: str = Field(..., description="LoRA filename (e.g., 'style.safetensors')")
    weight: float = Field(1.0, description="LoRA weight multiplier")


class LoadModelRequest(BaseModel):
    """Request to load a model."""
    model_type: str = Field(..., description="Model type identifier (e.g., 't2v', 'i2v')")
    profile: int = Field(-1, description="Memory profile (0-5), -1 for default")


class GenerationRequest(BaseModel):
    """Request for video/image generation."""

    # Required
    prompt: str = Field(..., description="Text prompt for generation")

    # Model selection (optional if model already loaded)
    model_type: Optional[str] = Field(None, description="Model type to use")
    model_def: Optional[ModelDefinition] = Field(None, description="Dynamic model definition (registers model at runtime)")
    job_id: Optional[str] = Field(None, description="Job ID for profiling correlation")

    # Core generation parameters
    resolution: str = Field("832x480", description="Output resolution (e.g., '1024x576')")
    video_length: int = Field(81, description="Number of frames to generate")
    num_inference_steps: int = Field(30, description="Number of denoising steps")
    seed: int = Field(-1, description="Random seed (-1 for random)")
    guidance_scale: float = Field(5.0, description="Classifier-free guidance scale")
    negative_prompt: str = Field("", description="Negative prompt")

    # Image inputs (base64 encoded or file paths)
    image_start: Optional[str] = Field(None, description="Start image (base64 or path)")
    image_end: Optional[str] = Field(None, description="End image (base64 or path)")
    image_refs: Optional[List[str]] = Field(None, description="Reference images")

    # Video inputs
    video_source: Optional[str] = Field(None, description="Source video path")
    video_guide: Optional[str] = Field(None, description="Guide video path")

    # Image/video mode
    image_mode: int = Field(0, description="0=video, 1=image, 2=inpaint")
    image_prompt_type: str = Field("", description="Image prompt type flags")
    video_prompt_type: str = Field("", description="Video prompt type flags")

    # Advanced generation settings
    activated_loras: List[str] = Field(default_factory=list, description="Active LoRA names")
    loras_multipliers: str = Field("", description="LoRA weight multipliers")
    flow_shift: float = Field(3.0, description="Flow shift parameter")
    sample_solver: str = Field("", description="Sampler solver type")
    embedded_guidance_scale: float = Field(6.0, description="Embedded guidance scale")

    # Guidance phases
    guidance_phases: int = Field(1, description="Number of guidance phases")
    guidance2_scale: float = Field(5.0, description="Phase 2 guidance scale")
    guidance3_scale: float = Field(5.0, description="Phase 3 guidance scale")
    switch_threshold: float = Field(0, description="Phase 1-2 switch threshold")
    switch_threshold2: float = Field(0, description="Phase 2-3 switch threshold")
    model_switch_phase: int = Field(1, description="Model switch phase")
    alt_guidance_scale: float = Field(6.0, description="Alternative guidance scale")

    # Sliding window for long videos
    sliding_window_size: int = Field(129, description="Sliding window size (0 to disable)")
    sliding_window_overlap: int = Field(5, description="Sliding window overlap frames")
    sliding_window_color_correction_strength: float = Field(0, description="Color correction")
    sliding_window_overlap_noise: float = Field(0, description="Overlap noise")
    sliding_window_discard_last_frames: int = Field(0, description="Frames to discard")

    # Post-processing
    temporal_upsampling: str = Field("", description="Temporal upsampling method")
    spatial_upsampling: str = Field("", description="Spatial upsampling method")
    film_grain_intensity: float = Field(0, description="Film grain intensity")
    film_grain_saturation: float = Field(0.5, description="Film grain saturation")

    # Audio
    MMAudio_setting: int = Field(0, description="MMAudio setting (0=off)")
    MMAudio_prompt: str = Field("", description="MMAudio prompt")
    MMAudio_neg_prompt: str = Field("", description="MMAudio negative prompt")
    audio_guidance_scale: float = Field(4.0, description="Audio guidance scale")
    audio_scale: float = Field(1.0, description="Audio scale")
    audio_prompt_type: str = Field("", description="Audio prompt type")

    # Control settings
    control_net_weight: float = Field(1.0, description="ControlNet weight")
    control_net_weight2: float = Field(1.0, description="ControlNet weight 2")
    control_net_weight_alt: float = Field(1.0, description="ControlNet alt weight")
    motion_amplitude: float = Field(1.0, description="Motion amplitude")
    denoising_strength: float = Field(0.5, description="Denoising strength")
    masking_strength: float = Field(1.0, description="Masking strength")
    input_video_strength: float = Field(1.0, description="Input video strength")

    # Advanced
    NAG_scale: float = Field(1.0, description="NAG scale")
    NAG_tau: float = Field(3.5, description="NAG tau")
    NAG_alpha: float = Field(0.5, description="NAG alpha")
    skip_steps_cache_type: str = Field("", description="Step skipping cache type")
    skip_steps_multiplier: float = Field(1.75, description="Step skip multiplier")
    skip_steps_start_step_perc: float = Field(0, description="Step skip start percentage")

    # Output
    batch_size: int = Field(1, description="Batch size")
    repeat_generation: int = Field(1, description="Number of repeats")
    override_profile: int = Field(-1, description="Override memory profile")
    output_filename: str = Field("", description="Custom output filename")
    force_fps: str = Field("", description="Force specific FPS")

    class Config:
        extra = "allow"  # Allow extra fields for forward compatibility
