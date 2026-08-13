"""ID-V2V worker entry point."""
import os

# Must be set before torch initializes the CUDA allocator. expandable_segments
# lets PyTorch grow/reuse memory segments instead of fragmenting, which closes
# the last ~0.5 GB gap needed to fit 720p/81-frame denoise in the 32 GB GPU
# alongside the resident fp8 DiT/VACE model.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from runner.idv2v.server import main

main()
