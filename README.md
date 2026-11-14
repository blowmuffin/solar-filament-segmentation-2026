# filseg: solar filament instance segmentation

Our solution to the *Solar Filament Segmentation Challenge 2026* (Kaggle competition `filament-segmentation-2026`, metric: Panoptic Quality, PQ). A U-Net predicts where filaments are; the connected pieces of its probability map become the instances; a small keep rule discards the weak ones. The repository holds the training code, the inference code, the final weights and one notebook that rebuilds the submission file from the weights without any training.

Author: Ayush Kasare. Licence: MIT (`LICENSE`).

## Results

Everything marked `TBD` is filled in from the final runs. PQ is the host's pooled Panoptic Quality: predicted and true filaments match at IoU above 0.5, `PQ = sum of IoU over matches / (TP + FP/2 + FN/2)`, `SQ` is the mean IoU of the matches and `RQ = TP / (TP + FP/2 + FN/2)`. The out-of-fold numbers come from the five-fold split below, with the keep rule of every fold tuned on the other four folds only (nested). Intervals are 95 % bootstrap intervals over images (2000 resamples, `filseg.tune.bootstrap_ci`).

| quantity | value |
|---|---|
| Out-of-fold PQ, nested, final submission | TBD (95 % interval TBD to TBD) |
| Out-of-fold SQ / RQ | TBD / TBD |
| No-training baseline (`filseg/baseline.py`), same protocol | TBD (95 % interval TBD to TBD) |
| Gain over the baseline, paired bootstrap | TBD (95 % interval TBD to TBD) |
| Kaggle public leaderboard PQ | TBD |
| Kaggle private leaderboard PQ | TBD |
| Working resolution (`work_res`, decided by a pilot) | TBD px |
| Reproduction: PQ of the notebook's CSV against the submitted CSV (target at least 0.99) | TBD on a T4, TBD on a CPU |
| Notebook run time, 180 test images | TBD min on a T4, TBD min on a CPU |
| Size of one weight file | TBD MB (fp16) |

## The pipeline in six lines

1. **Normalise.** Each image is z-scored with the mean and standard deviation of the pixels inside 0.98 of the solar radius; the sky outside the disk is set to 0.
2. **Network.** A U-Net with a ResNet34 encoder (ImageNet initialisation, one input channel) predicts one probability per pixel.
3. **Training.** The target is the mean of all annotators' masks of the image (a soft target), the loss is binary cross-entropy plus 0.5 times soft Dice, the inputs are 512 px crops (70 % of them centred on a filament) with flips, quarter turns and a small brightness jitter. AdamW, one warm-up epoch then cosine decay, 40 epochs of 250 steps, weights averaged over time (EMA). One model per fold, five folds, trained on Kaggle T4 GPUs.
4. **Predict.** Each test image is predicted whole at the working resolution, in four views (as is, left-right, up-down, both) that are flipped back and averaged, then averaged over the five fold models and brought to 2048 px.
5. **Instances.** The 8-connected components of the probability map above a low threshold are the candidates. A keep rule (peak and mean probability, area, and a limb cut on the distance from the disk centre as a fraction of the solar radius) discards weak candidates. The rule is tuned by nested five-fold cross-validation on the host-exact pooled PQ.
6. **Output.** The kept components become pixel-disjoint column-major RLE rows `<image stem>_<k>`, and the file is validated before it is written out.

## What is in the repository

| path | what it does |
|---|---|
| `notebook.ipynb` | The deliverable: runs the whole pipeline from the committed weights to `submission.csv`, with figures. |
| `filseg/data.py` | Where the data lives, reading the annotation file (its SHA-256 is checked), turning the polygons into masks, building the soft target (mean of the annotators), the normalisation, the frozen folds, the training cache. |
| `filseg/config.py`, `configs/runs.json` | Settings of each named training run (resolution, crop size, seed, ...). |
| `filseg/model.py` | Builds the U-Net, saves and loads the fp16 weight files, says where a run's files go. |
| `filseg/train.py` | Trains one fold: samples crops, computes the loss, keeps the EMA weights, saves them in fp16. Needs a GPU. |
| `filseg/predict.py` | Probability map of one image from flip views; its command-line entry predicts a fold's validation images and all test images on a GPU. |
| `filseg/job.py`, `tools/make_kernel.py` | Run fold jobs on Kaggle's two T4 GPUs; build the Kaggle notebook that carries the code (it refuses a modified git tree and records the commit). |
| `filseg/infer.py` | Reads `weights/manifest.json`, checks the weight files, builds the ensemble map and turns one image into masks; also writes the manifest. What the notebook calls. |
| `filseg/instances.py` | Components of a probability map, their features, the keep rule, RLE encoding, the CSV writer; its command-line entry builds a CSV from the prediction files of a run. |
| `filseg/preflight.py` | Submission checker (format, overlaps, row counts), pre-submission checks, preview sheet. |
| `filseg/pq.py` | The competition metric, re-implemented and checked against the host's own scorer. |
| `filseg/baseline.py` | A no-training baseline (dark structure on a brighter neighbourhood) run through the same rule and metric. |
| `filseg/tune.py` | Tunes the keep rule: candidate table, grid search, nested out-of-fold evaluation, bootstrap intervals, failure buckets. Writes `filseg/rule.json`. |
| `filseg/eda.py` | Data checks: site and year mix, train-test drift, annotator agreement, filament geometry. |
| `filseg/folds.json` | The frozen assignment of every training image to a fold. |
| `filseg/rule.json` | The final keep rule. |
| `weights/` | The final fp16 weights, one file per fold, and `manifest.json`. |
| `tests/` | One test file per module (`python -m pytest tests -q`). |

## Reproduce the submission

The notebook needs the test images, `weights/` and `filseg/rule.json`; nothing else. With the annotation file also present it scores one training fold at the end.

Settings are at the top of the notebook (`DATA_DIR`, `N_IMAGES`, `RUN_TRAINING`). Four environment variables can set the same things before launch: `FILSEG_DATA` (folder that holds `test/test_images` and, optionally, `train/`), `FILSEG_WEIGHTS` (default `weights`), `FILSEG_RULE` (default `filseg/rule.json`), `FILSEG_OUT` (where `submission.csv` is written, default the current folder). Relative paths are taken from the repository root.

**On Kaggle**

1. Create a notebook, switch the accelerator to a T4 GPU (a CPU works, slower) and add the competition data as an input; the notebook finds it under `/kaggle/input`.
2. Get the repository: `!git clone <URL of this repository>` and `%cd` into it, or attach it as a dataset.
3. `!pip install -q segmentation-models-pytorch==0.6.0 pycocotools==2.0.11` (the Kaggle image already has the other packages).
4. Run `notebook.ipynb`, or headless: `!FILSEG_OUT=/kaggle/working jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /kaggle/working`. The result is `submission.csv`.

**Locally**

```
git clone <URL of this repository>
cd <repository>
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt jupyterlab                # any recent Jupyter will do, it is not pinned
export FILSEG_DATA=/path/to/MAGFiLO_1.0_Kaggle_2026       # the folder that holds test/ (and train/)
jupyter lab notebook.ipynb
```

For a quick check on a machine without a GPU, set `N_IMAGES = 2` in the settings cell: it runs top to bottom in a few minutes. The notebook picks the GPU if there is one and falls back to the CPU. The two give nearly the same probability maps (the GPU runs the network in fp16 mixed precision, the CPU in fp32), so a component right at a threshold can differ; the reproduction row in the results table measures this.

## Weights and keep rule

`weights/manifest.json` lists the models the notebook loads, so the notebook needs no knowledge of file names or resolutions:

```json
{"models": [{"file": "A_fold0.pt", "run": "A", "fold": 0, "encoder": "resnet34",
             "work_res": 2048, "sha256": "...", "commit": "..."}, ...]}
```

`work_res` is the resolution the model was trained and predicts at (the image is area-resized to it and the map is brought back to 2048 px); `commit` is the code version that trained it. The weight files are fp16 `state_dict`s, each under 100 MB; `infer.load_models` checks every file against its SHA-256 and never downloads anything (the ImageNet initialisation is only needed for training and is overwritten by the trained weights). All models of the manifest are averaged with equal weight, exactly as `instances.py` averages the prediction files of a run, so the notebook and the command line build the same CSV from the same models.

`filseg/rule.json` is the keep rule written by `filseg.tune`: `t_lo` (component threshold), `peak` and `mean` (minimum peak and mean probability inside the component), `min_area` (pixels at 2048 px), `max_r` (largest allowed distance of the component centre from the disk centre, as a fraction of the solar radius).

## Retrain on Kaggle

Training runs on Kaggle's two T4 GPUs, one fold per GPU, so five folds take three kernels (folds 0 and 1, 2 and 3, 4). You need a Kaggle account with the competition rules accepted and the Kaggle CLI (`python -m kaggle`); change `OWNER` in `tools/make_kernel.py` to your user name. Run from the repository root, with a clean git tree (the builder refuses a modified one so that every run records one commit):

```
python tools/make_kernel.py --job retrain-a --jobs A:0 A:1
python -m kaggle kernels push -p ../kaggle/retrain-a --accelerator NvidiaTeslaT4 -t 14400
python tools/make_kernel.py --job retrain-b --jobs A:2 A:3
python -m kaggle kernels push -p ../kaggle/retrain-b --accelerator NvidiaTeslaT4 -t 14400
python tools/make_kernel.py --job retrain-c --jobs A:4
python -m kaggle kernels push -p ../kaggle/retrain-c --accelerator NvidiaTeslaT4 -t 14400
```

`A` is a run name from `configs/runs.json` (`A` is the default setting at 2048 px, `B` the same at 1024 px). Each kernel builds the training cache, trains, and predicts the fold's validation images and the test images. When a kernel has finished (`python -m kaggle kernels status <owner>/filseg-retrain-a`), download its files into `../runs` so that `../runs/A/fold<k>.pt` exists:

```
python -m kaggle kernels output <owner>/filseg-retrain-a -p ../runs --file-pattern <pattern>
```

Then, from the repository root:

```
python -m filseg.tune --run A --rule-out filseg/rule.json     # tune the keep rule on the out-of-fold maps
python -m filseg.infer --run A --out weights                  # copy the fold weights, write weights/manifest.json
python -m filseg.instances --run A --split test --rule filseg/rule.json --out submission.csv
```

The last command builds the CSV from the saved test maps instead of re-running the networks; the notebook builds it from the weights. The notebook's last cell prints the same training commands and can run them on a machine with GPUs (`RUN_TRAINING = True`).

## The split

707 training images, five folds of 141 or 142 images, seed 0, frozen in `filseg/folds.json` before any model was trained. The folds are grouped by image: an image can have up to three annotators' records, and all of them stay in the same fold, otherwise a model would be validated on a filament it has seen under another annotator's name. They are balanced by the number of annotator records and by site (the two-letter code that ends each file name; the file name is used for this balancing and for data analysis only, never as a model input). The 180 test images are never used for training or tuning. All tuning (keep rule, resolution, other choices) uses out-of-fold predictions only, and the reported rule is evaluated nested.

## Training environment

| | |
|---|---|
| Hardware | Kaggle batch kernels, two NVIDIA Tesla T4 (16 GB), one fold job per GPU, `-t 14400` (4 h) limit |
| Precision | fp16 autocast with a gradient scaler for training, fp16 autocast for GPU prediction, fp16 weight files |
| Kaggle image | `gcr.io/kaggle-gpu-images/python`; Python 3.13.16, torch 2.11.0+cu130, CUDA 13.0, segmentation-models-pytorch 0.6.0, timm 1.0.30, numpy 2.1.3, opencv-python-headless 4.14.0.94, pycocotools 2.0.11 (as recorded by the pilot job; every job writes its own `fold<k>_meta.json` with the commit, seed, versions and timings) |
| Seeds | `1000 x run seed + fold` |
| Local machine | Windows 11, Python 3.13.7, CPU only: tests and tuning, no training. `requirements.txt` pins the versions used there; the weights are plain tensors and load under both versions. |

## Tests

```
python -m pytest tests -q
```

One test file per module. Tests marked `slow` need files from outside the repository (the host's scorer notebook, the full data); leave them out with `-m "not slow"`. No test runs an optimiser step.

## Data use statement

- Only the competition's own data is used: the training images and their annotations, and the test images. The loader checks the SHA-256 of the annotation file. The test images are used only to predict.
- The only outside resource is the ImageNet-pretrained weights of the ResNet34 encoder (from `segmentation-models-pytorch`), used to initialise training. Inference does not need them.
- The public MAGFiLO release and its mirrors, the SWEFil data, external labels, other teams' weights or predictions, and any test-time metadata are not used anywhere in the pipeline. No competition image or label is stored in this repository.

## AI-assistance disclosure

The code, tests and documentation in this repository were written with the help of an AI coding assistant (Claude Code, Anthropic) under the author's direction. The author reviewed the code and the results and can explain every component.

## Licence and author

MIT, see `LICENSE`. Author: Ayush Kasare.
