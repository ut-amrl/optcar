"""Visible Torch devices shared by the GUI, generator, and environment setup."""
import torch


def available_devices():
    devices = [{"id": "cpu", "name": "CPU", "memory_gib": None}]
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append({"id": f"cuda:{index}", "name": properties.name,
                            "memory_gib": round(properties.total_memory / 2**30, 1)})
    return {"devices": devices, "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "default": devices[1]["id"] if len(devices) > 1 else "cpu"}


def resolve_device(device):
    """Validate before creating output files; retain Torch's indexed GPU routing."""
    try:
        selected = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"Invalid device: {device}") from exc
    if selected.type == "cpu":
        return "cpu"
    if selected.type != "cuda":
        raise ValueError("Choose cpu or an available cuda:N GPU")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Run bash scripts/setup_env.sh and restart the GUI; check the NVIDIA driver and GPU access.")
    index = selected.index if selected.index is not None else torch.cuda.current_device()
    count = torch.cuda.device_count()
    if not 0 <= index < count:
        raise ValueError(f"GPU cuda:{index} is unavailable; {count} GPU(s) are visible. Refresh the device list.")
    return f"cuda:{index}"
