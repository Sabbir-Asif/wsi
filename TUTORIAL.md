# Running ABMILX end to end: tutorial

This guide takes you from a clean machine to a trained model. It has three parts:

1. **Environment setup**: the versions that actually work.
2. **Smoke test**: run the pipeline on a single sample image in about a minute, to confirm the install.
3. **Full pipeline**: raw WSIs → patches (`.h5`) → LMDB → end-to-end training. A two-stage (pre-extracted features) variant is also covered.

Items marked ✅ were run and verified (macOS, CPU, Python 3.11). Items marked ⚠️ come from reading the code and were **not** executed (openslide preprocessing, multi-GPU training, feature-based training).

---

## 1. Environment setup

### 1.1 Prerequisites

| Need | Why |
|---|---|
| **Python 3.10 – 3.12** (✅ 3.11) | `torchaug>=0.4` requires Python `<3.13`. On 3.13 only `torchaug<=0.3.3` installs, and it lacks `SequentialTransform` / `RandomResizedCrop`, so training crashes. |
| `libjpeg-turbo` | `PyTurboJPEG` decodes patches with it. macOS: `brew install jpeg-turbo`. Ubuntu: `sudo apt install libturbojpeg`. |
| OpenSlide (only for raw-WSI preprocessing) | `CLAM/create_patches_fp.py` and `CLAM/save_patches.py` read slides with it. macOS: `brew install openslide`. Ubuntu: `sudo apt install openslide-tools`. |
| NVIDIA GPU + CUDA (for real training) | The device is chosen as `cuda` if available, else `cpu`. There is no Apple-GPU (MPS) path. CPU works for tiny tests only. |

### 1.2 Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate

# Core dependencies. Do NOT use plain `pip install -r requirements.txt`; see the note below.
pip install torch torchvision timm einops lmdb opencv-python "pandas<3" \
            scikit-learn torchmetrics h5py pillow numpy \
            jpeg4py PyTurboJPEG scikit-survival wandb \
            GitPython future fairscale

# torchaug must be 0.4+ (provides SequentialTransform, RandomResizedCrop).
# It pins an old torch/torchvision, so install it WITHOUT dependencies.
pip install torchaug==0.6.1 --no-deps
```

> ✅ I installed `torchaug==0.6.1` normally, which downgraded `torch` to 2.2.1. I then ran `pip install -U "torch>=2.5.1" torchvision` and the run worked with `torch 2.15.0`, `torchvision 0.29.2` and `torchaug 0.6.1`. pip prints a version-conflict warning for `torchaug`, which is harmless. `--no-deps` above is the cleaner equivalent, but I did not run that exact command.
> The README requires `torch >= 2.5.1`; `modules/vit_mil.py` imports `torch.nn.attention`, which older torch does not have.

**Why not `pip install -r requirements.txt`?**
- It omits `GitPython` (imported by `utils.py`) and `fairscale` (imported by `modules/longnet`). Both caused `ModuleNotFoundError`.
- It doesn't pin `torchaug`, and the default version for your Python may be wrong (see above).
- It doesn't pin `pandas`. **pandas 3.x breaks** `datasets/dataset_img.py` (`TypeError: Invalid value ... for dtype 'str'` when mapping labels). Use `pandas<3` (I had 2.3.3).

### 1.3 Pretrained encoder weights

`e2e_r18_*` / `e2e_r50_*` build the encoder with `timm.create_model(..., pretrained=True)` (`modules/encoders.py`, default `resnet50.tv_in1k`), so the first run **downloads ImageNet weights from the Hugging Face hub**. You need internet access, or a pre-populated `~/.cache/huggingface`.

### 1.4 Weights & Biases

`--wandb` is optional. Be aware: with `--wandb`, `utils.check_and_commit_changes` **runs `git add` on untracked files and `git commit -a -m "Run Auto commit"` in this repo**. If you don't want auto-commits, omit `--wandb`. To silence W&B entirely, set `WANDB_MODE=disabled`.

---

## 2. Smoke test with one sample image ✅

This verifies the install and the training/eval loop without any real data. It builds a tiny fake LMDB from `doc/abx.png`.

Training expects LMDB files with this layout (the exact format `CLAM/save_patches.py` writes):

| Key | Value |
|---|---|
| `{slide_id}-{i}` | `pickle.dumps(<JPEG bytes>)` |
| `__keys__` | pickled list of all patch keys (bytes) |
| `__len__` | pickled int, total patches |
| `__slide__` | pickled list of slide ids |
| `__pn__` | pickled dict `slide_id → patch count` |

Save as `make_sample.py` and run it:

```python
# python make_sample.py doc/abx.png sample_data
import io, os, pickle, random, sys
import lmdb
from PIL import Image

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
im = Image.open(src).convert("RGB"); W, H = im.size
random.seed(0)
slides = [f"SLIDE{i}" for i in range(8)]
PN = 16                                   # patches per slide

# Two scales (main + "sub" for multi-scale input), each at 224 and 256.
# "224" MUST appear in the file name; see the note on 224/256 below.
for scale in ("sample", "sub"):
    for size in (224, 256):
        db = lmdb.open(f"{out}/{scale}_{size}.lmdb", subdir=False, map_size=1 << 30)
        keys, pn = [], {}
        with db.begin(write=True) as txn:
            for s in slides:
                pn[s] = PN
                for i in range(PN):
                    x, y = random.randint(0, W - size), random.randint(0, H - size)
                    b = io.BytesIO()
                    im.crop((x, y, x + size, y + size)).save(b, format="JPEG", quality=90)
                    k = f"{s}-{i}".encode("ascii"); keys.append(k)
                    txn.put(k, pickle.dumps(b.getvalue()))
            txn.put(b"__keys__", pickle.dumps(keys))
            txn.put(b"__len__", pickle.dumps(len(keys)))
            txn.put(b"__slide__", pickle.dumps(slides))
            txn.put(b"__pn__", pickle.dumps(pn))
        db.close()

with open(f"{out}/labels.csv", "w") as f:       # 4 IDC + 4 ILC
    f.write("case,slide,label,fold\n")
    for i, s in enumerate(slides):
        f.write(f"C{i},{s},{'IDC' if i % 2 == 0 else 'ILC'},train\n")
```

Run training (`main.py` creates `<output_path>/<project>/<title>` with `os.mkdir`, **not** `makedirs`, so the `--output_path` directory must already exist):

```bash
mkdir -p sample_out
WANDB_MODE=disabled python main.py \
  --datasets=brca --project=smoke --title=smoke \
  --dataset_root=$PWD/sample_data/sample_224.lmdb \
  --dataset_sub_root=$PWD/sample_data/sub_224.lmdb \
  --csv_path=$PWD/sample_data/labels.csv \
  --model=e2e_r18_abmilx -c=config/e2e/r18_brca_subtyping.yaml \
  --output_path=$PWD/sample_out --lmdb \
  --cv_fold=2 --num_epoch=1 --max_epoch=1 \
  --num_workers=2 --batch_size=1 --same_psize=8 --p_batch_size=8
```

**Expected result** (about a minute on a laptop CPU): exit code 0, per-fold `Train: 0 [...]` lines, an `Epoch [1/1] train loss ... accuracy ... auc_value ...` line, `Cross validation accuracy mean ...` at the end, and four checkpoints in `sample_out/smoke/smoke/`: `fold_{0,1}_model_best.pt` and `fold_{0,1}_ema_model_best.pt`. Metrics will be around chance (~0.5) because this is 1 epoch on random crops.

Smoke-test gotchas I hit:
- `--num_workers 0` fails at the test stage with `lmdb.Error: ... is already open in this process`. Use `>= 1`.
- The config sets `mul_scale_input: true`, which requires `--dataset_sub_root`, pointing at a **different** LMDB file than `--dataset_root` (the same file also triggers the "already open" error).
- `--cv_fold=2` here makes the code build stratified k-folds from the CSV, ignoring its `fold` column. The shipped configs use `cv_fold: 1` (use the CSV's split column) or a folder of per-fold CSVs. See §3.4.

---

## 3. Full pipeline ⚠️

### 3.1 Option A: download the preprocessed LMDBs (recommended)

The authors provide ready-made LMDB tiles for every dataset:
- Hugging Face: https://huggingface.co/datasets/Dearcat/CPath_Image
- Baidu: https://pan.baidu.com/s/1H0P024gRIPgi3Xeu-xPLUg?pwd=t179 (password `t179`)

Skip to §3.4.

### 3.2 Option B: preprocess your own WSIs

Needs OpenSlide (§1.1). The CLAM scripts have their own imports (`wsi_core`, `utils`, `models`) and are run as `python CLAM/<script>.py` so Python finds them. Extra packages for CLAM are in `CLAM/env.yml` (`openslide-python`, `matplotlib`, `scipy`, `tqdm`, `PyYAML`, ...), and `CLAM/env.yml` pins `timm==0.9.8` (the e2e code works with timm 1.x, so use a single environment of your choice).

**Step 1: segment tissue and extract patch coordinates (`.h5`).**

Pick `--patch_size/--step_size/--patch_level` for the magnification you want (from the README):

| Dataset / magnification | `--patch_size` | `--step_size` | `--patch_level` |
|---|---|---|---|
| PANDA, ×40 | 256 | 256 | 0 |
| PANDA, ×20 | 512 | 512 | 0 |
| Other datasets, ×10 | 256 | 256 | 1 |
| Other datasets, ×5 | 512 | 512 | 1 |

```bash
python CLAM/create_patches_fp.py \
  --source /data/slides \
  --save_dir /data/clam_out \
  --patch_size 256 --step_size 256 --patch_level 1 \
  --preset CLAM/presets/tcga.csv \
  --seg --patch
```

Output: `/data/clam_out/patches/<slide_id>.h5` (coordinates) and `/data/clam_out/process_list_autogen.csv`. Presets are in `CLAM/presets/` (`tcga.csv`, `tcga_nsclc.csv`, `bwh_biopsy.csv`, `bwh_resection.csv`).

**Step 2: crop the patches and pack them into an LMDB.**

```bash
python CLAM/save_patches.py \
  --data_h5_dir /data/clam_out \
  --data_slide_dir /data/slides \
  --csv_path /data/clam_out/process_list_autogen.csv \
  --patch_dir /data/lmdb \
  --slide_ext .svs \
  --save_lmdb \
  --img_size 256 \
  --lmdb_name my_dataset_256_level1 \
  --batch_size 1024 --workers 6
```

Notes taken from `CLAM/save_patches.py`:
- `--data_h5_dir` is the folder that **contains** `patches/` (the script appends `patches/` itself).
- `--csv_path` must have a `slide_id` column that includes the file extension (as in `process_list_autogen.csv`). Only slides that have an `.h5` in `patches/` are processed.
- `--slide_ext` must match your files (`.svs`, `.tif`, ...).
- Output: `<patch_dir>/<lmdb_name>.lmdb` (JPEG quality set by `--img_quality`, default 80). Progress is logged to `<patch_dir>/<lmdb_name>_progress.txt`, so an interrupted run resumes.
- The LMDB map size is hard-coded to ~2 TB (sparse; fine on Linux/macOS, may fail on file systems without sparse-file support).
- **Run it twice**: once with `--img_size 256` (used for training) and once with `--img_size 224` (used for val/test).

### 3.3 The `224` / `256` naming rule (important)

`main.py` builds the training path by string replacement: when `img_size == 256` (all shipped e2e configs), it does

```python
args.env_train        = dataset_root.replace('224', '256')
args.dataset_sub_root = dataset_sub_root.replace('224', '256')
```

and val/test use `--dataset_root` as given. So:
- Pass `--dataset_root` / `--dataset_sub_root` pointing at the **224** LMDBs, with `224` in the file name.
- A same-named **256** LMDB must exist next to it (e.g. `foo_224_level1.lmdb` and `foo_256_level1.lmdb`).
- Don't let another `224` appear elsewhere in the path, or `replace` will rewrite it too.

### 3.4 Labels / splits

Pass `--csv_path`. Two forms are supported:

**Single CSV** with columns `case,slide,label,fold` (see `labels/panda.csv`, `labels/subtyping_tcga_brca.csv`). With `cv_fold: 1` in the config, the `fold` column (`train` / `val` / `test`) defines the split. With `cv_fold > 1` (CLI override) the CSV is stratified-k-fold split instead, ignoring `fold`.

**A directory of `fold_0.csv`, `fold_1.csv`, ...** (survival: `labels/survival_tcga_brca_3fold/`, columns `Study,ID,Event,Status,WSI,RNA,Label,Split`).

The `slide` column must match slide ids in the LMDB `__slide__` list (prefix matches are also accepted for classification). Label strings are hard-wired per `--datasets`:

| `--datasets` | Labels |
|---|---|
| `brca` | `IDC`, `ILC` |
| `nsclc` | `LUAD`, `LUSC` |
| `call` | `normal`, `tumor` |
| `panda` | integer ISUP grade |
| `surv_*` (e.g. `surv_brca`) | survival columns above |

### 3.5 End-to-end training (the main method)

Single GPU:

```bash
mkdir -p results      # must exist, see the smoke-test note
bash train.sh \
  --datasets=brca --project=MyProject --title=r18_brca_e2e_abmilx \
  --dataset_root=/data/lmdb/brca_224_level1.lmdb \
  --dataset_sub_root=/data/lmdb/brca_224_level0.lmdb \
  --csv_path=labels/subtyping_tcga_brca_resample.csv \
  --model=e2e_r18_abmilx \
  -c=config/e2e/r18_brca_subtyping.yaml \
  --output_path=results/brca_subtyping_e2e_r18_abmilx
```

Multi-GPU (`torchrun`, N processes):

```bash
bash train_dist.sh 2 --datasets=panda --project=MyProject --title=r18_panda_e2e_abmilx \
  --dataset_root=$LMDB_224 --dataset_sub_root=$LMDB_SUB_224 \
  --csv_path=labels/panda.csv --model=e2e_r18_abmilx \
  --output_path=results/panda_e2e_r18_abmilx -c=config/e2e/r18_panda.yaml
```

Notes:
- `train.sh` sets `CUBLAS_WORKSPACE_CONFIG` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and calls `python3 main.py "$@"`.
- Configs: `config/e2e/{r18_panda, r50_panda, r18_brca_subtyping, r50_brca_subtyping, r18_brca_survival}.yaml`. CLI flags override YAML values.
- `--model=e2e_{encoder}_{mil}`, e.g. `e2e_r18_abmilx`, `e2e_r50_abmilx`.
- **Multi-scale input**: all shipped e2e configs set `mul_scale_input: true`, which needs `--dataset_sub_root` (a second LMDB, e.g. a different magnification). The README's example commands pass `--dataset_sub_root=$PATH_TO_LOW_LEVEL_DATA` and `--dataset_root=$PATH_TO_HIGH_LEVEL_DATA`. If you have only one scale, set `mul_scale_input: false` in a copy of the config (I did not test single-scale training).
- Memory: `same_psize` (patches per slide per step), `p_batch_size` (encoder chunk size) and `batch_size` control GPU memory. Lower them if you hit OOM. The smoke test used `8 / 8 / 1`; the configs use e.g. `512 / 512 / 3` (BRCA) and `128 / 2048 / 8` (PANDA).
- Outputs go to `<output_path>/<project>/<title>/`: `fold_{k}_model_best.pt`, `fold_{k}_ema_model_best.pt`, and the printed cross-validation metrics.

### 3.6 Two-stage framework (pre-extracted features)

1. Get features: download from https://huggingface.co/datasets/Dearcat/CPathPatchFeature (or Baidu link in the README), or extract your own with [TRIDENT](https://github.com/mahmoodlab/TRIDENT) or CLAM's `extract_features_fp.py`.
2. The feature loader expects `--dataset_root` to contain a **`pt_files/`** subfolder with one `.pt` file per slide (`datasets/dataset_feat.py`). Optional `--h5_path` points at coordinate `.h5` files for position encodings.
3. Train a MIL head:

```bash
bash train.sh --datasets=panda --project=MyProject --title=panda_feat_abmil \
  --dataset_root=/data/panda_features --csv_path=labels/panda.csv \
  --model=abmil -c=config/feat/fm_panda.yaml \
  --output_path=results/panda_feat_abmil
```

Configs: `config/feat/{fm_panda, fm_subtyping, fm_survival}.yaml`. Other MIL models live in `modules/` (`abmil`, `abmilx`, `clam`, `dsmil`, `dtfd`, `transmil`, `wikg`, ...).

To use ABMILX as a drop-in head in your own code:

```python
from modules.abmilx import DAttentionX
head = DAttentionX(input_dim=1024, n_classes=2, mil_bias=True, attn_bias=True, n_heads=8)
# bag of features: (1, N, D)  ->  logits (1, n_classes)
```

---

## 4. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: git` | `pip install GitPython` |
| `ModuleNotFoundError: future` | `pip install future` |
| `ModuleNotFoundError: fairscale` | `pip install fairscale` |
| `No module named 'torch.nn.attention'` | torch is too old (often downgraded by installing `torchaug`). Reinstall `torch>=2.5.1`. |
| `module 'torchaug.transforms' has no attribute 'RandomResizedCrop'` | Wrong `torchaug` (Python 3.13 gets ≤0.3.3). Use Python ≤3.12 and `torchaug>=0.4`. |
| `TypeError: Invalid value '[...]' for dtype 'str'` | pandas 3.x. `pip install "pandas<3"`. |
| `AttributeError: 'NoneType' object has no attribute 'begin'` | `mul_scale_input: true` but no `--dataset_sub_root`. |
| `lmdb.Error: ... is already open in this process` | `--num_workers 0`, or the root and sub-root are the same file. |
| `FileNotFoundError` on `.../smoke` or the project dir | `--output_path` must already exist (`os.mkdir`, not `makedirs`). |
| `lmdb.Error` / not found on a `*_256*` path | The 256 LMDB twin doesn't exist; see §3.3. |
| Runs on CPU although you have a GPU | `torch.cuda.is_available()` is false: check your CUDA build of torch. |
