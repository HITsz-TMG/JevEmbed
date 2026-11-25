"""Shared device and dtype selection for local model backends."""

from ..errors import BackendError, ValidationError


def resolve_device(config, torch):
    device = ("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else config.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise BackendError("CUDA was requested but is unavailable")
    return device


def resolve_device_dtype(config, torch, *, reject_float16=False):
    device = resolve_device(config, torch)
    dtype = config.dtype
    if dtype == "auto":
        dtype = "bfloat16" if device.startswith("cuda") and torch.cuda.is_bf16_supported() else "float32"
    if reject_float16 and dtype == "float16":
        raise ValidationError("This model can return NaNs in float16; use bfloat16 or float32")
    if device == "cpu" and dtype != "float32":
        raise ValidationError("CPU backend requires dtype=float32 or auto")
    return device, dtype, getattr(torch, dtype)
