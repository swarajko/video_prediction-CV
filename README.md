# LMC-Memory — Quick Usage

Minimal instructions to run inference and evaluation with the provided codebase.

Prerequisites
- Python 3.8+ (recommended)
- Install requirements:

```powershell
python -m pip install -r requirements.txt
```

Extract frames from .avi (no crop)

```powershell
python scripts/extract_kth_frames.py --videos_dir "path\to\avis" --out_dir "path\to\extracted_frames_root"
```

Run inference and evaluation

```powershell
python test.py --dataset kth --test_data_dir "path\to\extracted_frames_root" --checkpoint_load_file "checkpoints/modded_kth.pt" --test_result_dir "test_results/out" --img_size 128 --workers 0 --batch_size 1 --make_frame True --evaluate True
```

Quick notes
- Place checkpoints inside `checkpoints/` or give the full path to `--checkpoint_load_file`.
- Set `--img_size` to the size used by your checkpoint (the dataloader will resize input frames otherwise).
- For GPU acceleration, install a CUDA-enabled PyTorch version; otherwise CPU inference is used.

Thats it — the above steps are enough to extract frames, run the model, and evaluate results.
