# Open-Set Recognition of Plant Species

ISyE 6740 term project, Group 016 (Aswath Oruganti, Bakiye Hilal Khaniyev,
Yussif Salama).

We train image classifiers on a curated subset of Pl@ntNet-300K while holding
out 45 species entirely, then compare seven methods for deciding whether a test
image belongs to a species the model has never seen.

## Layout

| Folder | Contents |
| --- | --- |
| `Solution/` | All code, model checkpoints, cached embeddings, and figures |
| `report/` | LaTeX source for the final report and the proposal |
| `submissions/` | The PDFs actually handed in on Canvas |
| `course/` | Instructor-provided requirements and template — do not edit |
| `literature/` | Background papers (the Pl@ntNet-300K dataset paper) |
| `archive/` | Superseded drafts, kept for reference |

## Data

The dataset is not in version control. Download Pl@ntNet-300K and extract it so
that the images land at `Solution/Data/plantnet_300K/images/{train,val,test}/`.
The three metadata JSON files in that directory *are* tracked, via Git LFS.

## Pipeline

Every script under `Solution/` resolves its input paths relative to the current
working directory, so **run them from inside `Solution/`**, not from the project
root.

```bash
cd Solution

# 1. Curate: filter to species with >=150 training images, cap each at 300,
#    then split 181 known / 45 unknown at the species level (seed 42).
python biodiversity_screening_data_curation.py

# 2. Train a backbone on the known species. Each script fine-tunes, saves a
#    .pth checkpoint, and writes the representation-analysis figures.
python main_cnn_analysis.py              # ResNet-18, 5 epochs
python main_efficientnetB0_analysis.py   # EfficientNet-B0, 10 epochs

# 3. Extract and cache embeddings to open_set_outputs/*.npz.
python generate_open_set_embeddings.py --model resnet18
python generate_open_set_embeddings.py --model efficientnet

# 4. Score the seven detectors; writes the ROC figures and the headline table.
python open_set_scoring.py --model resnet18
python open_set_scoring.py --model efficientnet

# 5. Extended evaluation: closed-set metrics, taxonomic split, species-level
#    bootstrap, openness sweep. Caches per-sample scores so reruns are cheap.
python extended_evaluation.py --model both

# 6. Ablations diagnosing the one-class failure.
python oneclass_ablation.py --model both
```

Steps 1 through 3 are expensive: training took about 385 minutes for ResNet-18
and about 6,780 minutes for EfficientNet-B0. Steps 4 through 6 run on the cached
embeddings in minutes, dominated by the One-Class SVM fit.

Cached `.npz` score files are gitignored and regenerate deterministically.

## Building the report

```bash
cd report
latexmk -pdf -outdir=build final_report.tex
```

Figures are pulled from `../Solution/` and `../Solution/open_set_outputs/` via
`\graphicspath`, so the build must run from inside `report/`.

## Environment

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

The pinned packages cover steps 4 through 6. Steps 1 through 3 additionally
need torchvision, pandas, seaborn, and tqdm.
