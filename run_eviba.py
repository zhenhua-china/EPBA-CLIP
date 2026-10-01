import warnings
warnings.filterwarnings("ignore")

import csv
import os
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import torch
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image

from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast
from scripts.plot import visualize_vandt_heatmap
from eviba import eviba
import random
import numpy as np
import torch

def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(0)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLIP_INPUT_SIZE = 224
IMAGE_PREPROCESS_MODE = "resize_stretch"  
# "resize_stretch" keeps the full image; "center_crop" uses CLIP default.
TOP_K = 5



clip_path = r"models\clip-vit-base-patch32"

model = CLIPModel.from_pretrained(
    clip_path,
    local_files_only=True
).to(device)

processor = CLIPProcessor.from_pretrained(
    clip_path,
    local_files_only=True
)

tokenizer = CLIPTokenizerFast.from_pretrained(
    clip_path,
    local_files_only=True
)




def normalize_np(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.min()
    denom = x.max()
    if denom < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return x / denom


def prepare_clip_image(image):
    if IMAGE_PREPROCESS_MODE == "center_crop":
        return image

    if IMAGE_PREPROCESS_MODE != "resize_stretch":
        raise ValueError(f"Unsupported IMAGE_PREPROCESS_MODE: {IMAGE_PREPROCESS_MODE}")
    return image.resize((CLIP_INPUT_SIZE, CLIP_INPUT_SIZE), Image.BICUBIC)



def make_separate_save_paths(save_path):
    root, ext = os.path.splitext(save_path)
    ext = ext or ".png"
    return {
        "image_heatmap": f"{root}_image_heatmap{ext}",
        "text_heatmap": f"{root}_text_heatmap{ext}",
        "original_box": f"{root}_original_box{ext}",
        "plain_text": f"{root}_plain_text{ext}",
    }


def make_layer_diagnostics_path(save_path):
    root, _ = os.path.splitext(save_path)
    return f"{root}_layer_weights.csv"


def print_layer_diagnostics(branch_name, detail):
    layers = detail["layers"]
    scores = detail.get("scores")
    weights = detail["weights"]
    infos = detail["infos"]
    selected_layers = set(detail.get("selected_layers", []))

    print(f"{branch_name} selected layers: {sorted(selected_layers)}")
    print(f"{branch_name} layer diagnostics:")
    print("layer | sim_drop | KL | score | weight | selected")
    for idx, info in enumerate(infos):
        layer = layers[idx]
        score = float(scores[idx]) if scores is not None else 0.0
        weight = float(weights[idx])
        selected = "*" if layer in selected_layers else ""
        print(
            f"{layer:>5} | "
            f"{float(info['sim_drop']):.8f} | "
            f"{float(info['kl']):.8f} | "
            f"{score:.8f} | "
            f"{weight:.8f} | "
            f"{selected}"
        )


def save_layer_diagnostics(save_path, details):
    diag_path = make_layer_diagnostics_path(save_path)
    ensure_parent_dir(diag_path)

    with open(diag_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "branch",
            "top_k",
            "layer",
            "sim_drop",
            "kl",
            "score",
            "weight",
            "selected",
        ])

        for branch_key, branch_name in (("vision", "vision"), ("text", "text")):
            detail = details[branch_key][0]
            layers = detail["layers"]
            scores = detail.get("scores")
            weights = detail["weights"]
            infos = detail["infos"]
            selected_layers = set(detail.get("selected_layers", []))
            top_k = detail.get("top_k", "")

            for idx, info in enumerate(infos):
                layer = layers[idx]
                score = float(scores[idx]) if scores is not None else 0.0
                writer.writerow([
                    branch_name,
                    top_k,
                    layer,
                    float(info["sim_drop"]),
                    float(info["kl"]),
                    score,
                    float(weights[idx]),
                    int(layer in selected_layers),
                ])

    print(f"[run_eviba] layer diagnostics: {diag_path}")


def ensure_parent_dir(path):
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)


def save_fig(fig, path, tight=True, dpi=300):
    ensure_parent_dir(path)
    if tight:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    else:
        fig.savefig(path, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def image_to_float_array(image):
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    image = np.asarray(image).astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return image


def vmap_to_numpy(vmap):
    if torch.is_tensor(vmap):
        vmap = vmap.detach().cpu().numpy()
    return normalize_np(np.squeeze(vmap))


def resize_vmap_to_image(vmap, image):
    image_w, image_h = image.size if isinstance(image, Image.Image) else (image.shape[1], image.shape[0])
    vmap_img = Image.fromarray((vmap_to_numpy(vmap) * 255).astype(np.uint8))
    vmap_img = vmap_img.resize((image_w, image_h), Image.BICUBIC)
    return np.asarray(vmap_img).astype(np.float32) / 255.0


def add_bboxes(ax, bb, linewidth=2):
    if bb:
        for x, y, w, h in bb:
            rect = mpl.patches.Rectangle((x, y), w, h, linewidth=linewidth, edgecolor="r", facecolor="none")
            ax.add_patch(rect)


def save_image_panel(path, image, vmap=None, bb=None):
    image_arr = image_to_float_array(image)
    image_h, image_w = image_arr.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(figsize=(image_w / dpi, image_h / dpi), dpi=dpi)
    if vmap is not None:
        vmap_arr = vmap_to_numpy(vmap)
        if vmap_arr.shape != image_arr.shape[:2]:
            vmap_arr = resize_vmap_to_image(vmap_arr, image_arr)
        image_arr = show_cam_on_image(np.float32(image_arr), vmap_arr, use_rgb=True)
    ax.imshow(image_arr)
    add_bboxes(ax, bb, linewidth=max(2.0, min(image_w, image_h) / 280.0))
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save_fig(fig, path, tight=False, dpi=dpi)


def clean_text_tokens(text_words):
    return [token.split("<")[0] for token in text_words[1:-1]]


def align_text_scores(tmap, text_words):
    if torch.is_tensor(tmap):
        tmap = tmap.detach().cpu().numpy()
    text_scores = np.asarray(tmap).squeeze()
    if text_scores.shape[0] > len(text_words):
        text_scores = text_scores[: len(text_words)]
    elif text_scores.shape[0] < len(text_words):
        pad = len(text_words) - text_scores.shape[0]
        text_scores = np.pad(text_scores, (0, pad), mode="constant")
    return text_scores


def wrap_text_tokens(tokens, scores=None, max_chars=26):
    rows = []
    current = []
    current_scores = []
    current_len = 0

    for idx, token in enumerate(tokens):
        token_len = len(token)
        next_len = token_len if not current else current_len + 1 + token_len
        if current and next_len > max_chars:
            rows.append((current, current_scores))
            current = []
            current_scores = []
            current_len = 0

        current.append(token)
        if scores is not None:
            current_scores.append(float(scores[idx]))
        current_len = token_len if current_len == 0 else current_len + 1 + token_len

    if current:
        rows.append((current, current_scores))
    return rows


def plot_wrapped_text(ax, tokens, scores=None, max_chars=26, fontsize=30, left_pad_px=8, top_pad_px=8, word_gap_px=42):
    if scores is not None:
        scores = normalize_np(scores)

    rows = wrap_text_tokens(tokens, scores=scores, max_chars=max_chars)
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    line_height_px = fontsize * fig.dpi / 72.0 * 1.22
    y = top_pad_px

    for row_tokens, row_scores in rows:
        x = left_pad_px
        for idx, token in enumerate(row_tokens):
            kwargs = {"fontsize": fontsize, "color": "black", "va": "top"}
            if scores is not None:
                score = row_scores[idx]
                alpha = 0.15 + 0.75 * float(score)
                color = (0.2, 0.4, 1.0, alpha)
                kwargs["bbox"] = dict(facecolor=color, edgecolor=color, boxstyle="round,pad=0.06", alpha=alpha)
            text_obj = ax.text(x, y, token, **kwargs)
            fig.canvas.draw()
            bbox = text_obj.get_window_extent(renderer=renderer)
            x += bbox.width + word_gap_px
        y += line_height_px

    ax.axis("off")


def save_text_panel(path, tokens, scores=None):
    max_chars = 26
    fontsize = 30
    dpi = 300
    display_scores = normalize_np(scores) if scores is not None else None
    rows = wrap_text_tokens(tokens, scores=display_scores, max_chars=max_chars)
    line_height_px = fontsize * dpi / 72.0 * 1.22
    width_px = 1250
    height_px = max(190, int(16 + line_height_px * max(1, len(rows))))
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax.set_xlim(0, width_px)
    ax.set_ylim(height_px, 0)
    plot_wrapped_text(ax, tokens, scores=display_scores, max_chars=max_chars, fontsize=fontsize)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save_fig(fig, path, tight=True, dpi=dpi)


def save_separate_figures(save_path, image, vmap, text_words, text_scores, bb=None):
    paths = make_separate_save_paths(save_path)
    tokens = clean_text_tokens(text_words)
    token_scores = text_scores[1:-1]

    save_image_panel(paths["image_heatmap"], image=image, vmap=vmap, bb=bb)
    save_text_panel(paths["text_heatmap"], tokens=tokens, scores=token_scores)
    save_image_panel(paths["original_box"], image=image, vmap=None, bb=bb)
    save_text_panel(paths["plain_text"], tokens=tokens, scores=None)



def plot_result(image_path, text, vmap, tmap, save_path=None):
    image = Image.open(image_path).convert("RGB")
    image_for_clip = prepare_clip_image(image)

    text_ids = torch.tensor(
        [tokenizer.encode(text, add_special_tokens=True)]
    ).to(device)

    text_words = tokenizer.convert_ids_to_tokens(text_ids[0].tolist())

    image_under = processor(
        images=image_for_clip,
        return_tensors="pt",
        do_normalize=False
    )["pixel_values"][0].permute(1, 2, 0)

    if save_path is not None:
        ensure_parent_dir(save_path)

    visualize_vandt_heatmap(
        tmap,
        vmap,
        text_words,
        image_under,
        bb=None,
        title=save_path
    )


    if save_path is not None:
        text_scores = align_text_scores(tmap, text_words)
        save_separate_figures(
            save_path=save_path,
            image=image_under,
            vmap=vmap,
            text_words=text_words,
            text_scores=text_scores,
            bb=None,
        )





def run_eviba(image_path, text, save_path=None):
    image = Image.open(image_path).convert("RGB")
    image_for_clip = prepare_clip_image(image)

    image_feat = processor(
        images=image_for_clip,
        return_tensors="pt"
    )["pixel_values"].to(device)

    text_ids = torch.tensor(
        [tokenizer.encode(text, add_special_tokens=True)]
    ).to(device)

    
    vmap, tmap, details = eviba(
        model=model,
        text_ids=[text_ids],
        image_feat=image_feat,

        
        layers=tuple(range(12)),

       
        beta=0.1,
        var=1,
        lr=1,
        train_steps=10,

        
        noise_mode="patchwise",
        sigma_fixed=0.1,
        sigma_min=0.05,
        sigma_max=1.0,

        
        gamma_init=0.5,
        learnable_gamma=True,
        residual_mode="centered",

        
        temperature=0.5,
        kl_tau=1.0,
        top_k=TOP_K,

        return_details=True
    )

    
    print_layer_diagnostics("Vision", details["vision"][0])
    print_layer_diagnostics("Text", details["text"][0])
    if save_path is not None:
        save_layer_diagnostics(save_path, details)

    
    plot_result(
        image_path=image_path,
        text=text,
        vmap=vmap.squeeze(),
        tmap=tmap[0],
        save_path=save_path
    )

    return vmap, tmap, details




if __name__ == "__main__":
    img_path = ""
    text = ""
    save_path = f""


    
    print(f"[run_eviba] image: {img_path}")
    print(f"[run_eviba] text: {text}")
    print(f"[run_eviba] image_preprocess: {IMAGE_PREPROCESS_MODE}")
    print(f"[run_eviba] clip input: {CLIP_INPUT_SIZE}x{CLIP_INPUT_SIZE}")
    print(f"[run_eviba] top_k: {TOP_K}")
    print(f"[run_eviba] output: {save_path}")
    for name, path in make_separate_save_paths(save_path).items():
        print(f"[run_eviba] {name}: {path}")
    run_eviba(
        image_path=img_path,
        text=text,
        save_path=save_path
    )

