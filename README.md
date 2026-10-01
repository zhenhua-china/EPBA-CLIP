

## Code Structure

```text
.
├── eviba.py                         # Core EviBA implementation
├── run_eviba.py                     # Single-image visualization and Top-K diagnostics
├── evaluate_eviba_metrics.py        # Quantitative evaluation for EviBA
├── generate_attributions.py         # Baseline attribution generation/evaluation entry
├── demo.py                          # Single-image baseline visualization entry
├── AdaptiveNoiseGenerator.py        # Patch-wise adaptive noise module
├── datasets.py                      # CC, ImageNet, and Flickr8k dataset loaders
├── Flickr8k.py                      # Utility to build Flickr8k validation annotations
├── make_cc_csv.py                   # Utility to build Conceptual Captions CSV
├── salicncy/                        # Baseline attribution methods
└── scripts/                         # IBA, plotting, metrics, and utility modules
```

The `image/` directory contains a few lightweight example images for quick visualization. Full datasets, pretrained CLIP checkpoints, generated saliency maps, and large RISE masks are not included in this supplementary package.

## Environment

The code was tested with Python 3.8+ and PyTorch. Install dependencies with:

```bash
pip install -r requirements.txt
```

The code uses a frozen CLIP ViT-B/32 backbone. By default, scripts expect the HuggingFace CLIP checkpoint at:

```text
models/clip-vit-base-patch32/
```

If the checkpoint is not already available, download `openai/clip-vit-base-patch32` and save it to the path above. For example:

```python
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast

model_id = "openai/clip-vit-base-patch32"
save_dir = "models/clip-vit-base-patch32"

CLIPModel.from_pretrained(model_id).save_pretrained(save_dir)
CLIPProcessor.from_pretrained(model_id).save_pretrained(save_dir)
CLIPTokenizerFast.from_pretrained(model_id).save_pretrained(save_dir)
```

Natural-image EviBA scripts use `local_files_only=True` when loading CLIP, so the HuggingFace checkpoint must be prepared before running experiments. Medical scripts load a local medical CLIP checkpoint with `torch.load`.

## Dataset Preparation

The evaluation scripts support Conceptual Captions, ImageNet-style class-prompt evaluation, and Flickr8k. In this package, `tiny-imagenet-200/` is used as a lightweight ImageNet-style example directory. The expected directory layout is:

```text
datasets/
├── cc.csv
├── cc_images/
├── cc_txt/
├── en_val.json
├── Flickr8k/
│   ├── Flicker8k_Dataset/
│   └── Flickr8k_text/
└── tiny-imagenet-200/
```

For Flickr8k, `en_val.json` can be generated with:

```bash
python Flickr8k.py
```

For Conceptual Captions, `cc.csv` can be generated with:

```bash
python make_cc_csv.py
```

Dataset files are not redistributed in this package. Please obtain them from the corresponding official sources and place them under `datasets/`.

## Running EviBA

For a single image-text pair visualization:

```bash
python run_eviba.py
```

The default configuration in `run_eviba.py` uses:

```text
CLIP input size: 224 x 224
candidate layers: 0-11
Top-K: 5
beta: 0.1
variance: 1
learning rate: 1
bottleneck optimization steps: 10
noise mode: patchwise
sigma_fixed: 0.1
sigma_min: 0.05
sigma_max: 1.0
gamma_init: 0.5
residual mode: centered
KL exponent: 1.0
```

Outputs are written to `results/` and include the fused image heatmap, text token heatmap, plain text panel, original image panel, and layer diagnostics CSV.

For quantitative evaluation:

```bash
python evaluate_eviba_metrics.py --dataset flickr8k --layers 0-11 --beta 0.1 --train_steps 10
```

Useful arguments include:

```bash
--dataset {cc,imagenet,flickr8k}
--clip_path models/clip-vit-base-patch32
--batch_size 1
--max_samples 100
--layers 0-11
--beta 0.1
--var 1.0
--lr 1.0
--train_steps 10
--noise_mode patchwise
--sigma_fixed 0.1
--sigma_min 0.05
--sigma_max 1.0
--gamma_init 0.5
--residual_mode centered
--temperature 0.5
--kl_tau 1.0
```

## Baseline Methods

Baseline methods are implemented in `salicncy/` and can be run through `generate_attributions.py`:

```bash
python generate_attributions.py --dataset flickr8k --method gradcam --target_layer 9
python generate_attributions.py --dataset flickr8k --method saliencymap --target_layer 9
python generate_attributions.py --dataset flickr8k --method fast_ig --target_layer 9
python generate_attributions.py --dataset flickr8k --method chefer
python generate_attributions.py --dataset flickr8k --method mfaba
python generate_attributions.py --dataset flickr8k --method rise
python generate_attributions.py --dataset flickr8k --method m2ib --beta 0.1
python generate_attributions.py --dataset flickr8k --method nib --num_steps 10 --target_layer 9
```

Default baseline settings follow the implementation:

```text
Grad-CAM: target layer 9
Saliency Map: target layer 9
Fast-IG: target layer 9
Chefer: final attention block, no additional tunable hyperparameters
MFABA: 10 perturbation iterations, step size 0.01
RISE image branch: 6000 masks, grid size 8, mask probability 0.1, GPU batch 20
RISE text branch: 200 masks, grid size 8, mask probability 0.1
M2IB: beta 0.1, target layer 9, variance 1
NIB: 10 integration steps, target layer 9
```

## Top-K Ablation

To evaluate different Top-K values, modify `TOP_K` in `run_eviba.py`:

```python
TOP_K = 3  # or 5, 7, 9
```

Then run:

```bash
python run_eviba.py
```

The output filename includes the selected Top-K value, and the corresponding layer weights are saved in a CSV file.


## Medical CLIP Experiments

Medical visualization scripts are also included:

```text
run_medical_eviba.py   # Medical EviBA visualization with a medical CLIP checkpoint
run_medical.py         # Medical baseline visualization script
```

These scripts expect the medical CLIP checkpoint at:

```text
clip-imp-pretrained_128_6_after_4.pt
```

The medical CLIP checkpoint is not redistributed in this package due to size and data-usage restrictions. Please place the checkpoint in the project root and prepare permitted medical images locally before running:

```bash
python run_medical_eviba.py
python run_medical.py
```

## Notes

- The CLIP encoders are frozen for all experiments.
- All saliency maps are normalized to `[0, 1]` before visualization and metric evaluation.
- Large files such as pretrained checkpoints, full datasets, generated results, `masks.npy`, and Python cache files are intentionally excluded.
- If CUDA is unavailable, some baseline implementations may require minor edits because they explicitly call `.cuda()`.


