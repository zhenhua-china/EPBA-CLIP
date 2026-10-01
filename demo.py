import warnings
warnings.filterwarnings('ignore')
import os
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import clip
from scripts.plot import visualize_vandt_heatmap
from salicncy import chefer,fast_ig,gradcam,m2ib,mfaba,nib,rise,saliencymap
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast
from PIL import Image, ImageOps
from pytorch_grad_cam.utils.image import show_cam_on_image
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_path = r"models\clip-vit-base-patch32"

model = CLIPModel.from_pretrained(clip_path, local_files_only=True).to(device)
processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=True)
tokenizer = CLIPTokenizerFast.from_pretrained(clip_path, local_files_only=True)



OUTPUT_DIR = "results"
SAVE_SEPARATE = True
M2IB_RATE = 0.1
M2IB_LAYER = 9
NIB_STEPS = 10
NIB_LAYER = 9
GRADCAM_TARGET_LAYER = 6
ATTENTION_TARGET_LAYER = -1
CLIP_INPUT_SIZE = 224
IMAGE_PREPROCESS_MODE = "resize_stretch"  # "resize_stretch" keeps the full image; "center_crop" uses CLIP default.


def get_method_name(method):
    return getattr(method, "__name__", str(method))


def build_save_path(img_path, method, output_dir=OUTPUT_DIR):
    image_name = os.path.splitext(os.path.basename(img_path))[0]
    method_name = get_method_name(method)
    return os.path.join(output_dir, f"{image_name}_{method_name}.png")


def print_run_config(img_path, text, method, save_path, bb=None, save_separate=True):
    method_name = get_method_name(method)
    print(f"[demo] device: {device}")
    print(f"[demo] model: {clip_path}")
    print(f"[demo] image: {img_path}")
    print(f"[demo] text: {text}")
    print(f"[demo] method: {method_name}")
    if method_name == "m2ib":
        print(f"[demo] m2ib rate: {M2IB_RATE}")
        print(f"[demo] m2ib layer: {M2IB_LAYER}")
    elif method_name == "nib":
        print(f"[demo] nib steps: {NIB_STEPS}")
        print(f"[demo] nib layer: {NIB_LAYER}")
    elif method_name == "gradcam":
        print(f"[demo] gradcam target_layer: {GRADCAM_TARGET_LAYER}")
    elif method_name in {"fast_ig", "saliencymap"}:
        print(f"[demo] attention target_layer: {ATTENTION_TARGET_LAYER}")
    elif method_name == "mfaba":
        print("[demo] mfaba layer: full embedding perturbation, no target_layer")
    print(f"[demo] image_preprocess: {IMAGE_PREPROCESS_MODE}")
    print(f"[demo] clip input: {CLIP_INPUT_SIZE}x{CLIP_INPUT_SIZE}")
    print(f"[demo] save_separate: {save_separate}")
    print(f"[demo] bbox: {bb}")
    print(f"[demo] combined output: {save_path}")
    if save_separate:
        for name, path in make_separate_save_paths(save_path).items():
            print(f"[demo] {name}: {path}")


def normalize_np(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.min()
    denom = x.max()
    if denom < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return x / denom



def make_separate_save_paths(save_path):
    root, ext = os.path.splitext(save_path)
    ext = ext or ".png"
    return {
        "image_heatmap": f"{root}_image_heatmap{ext}",
        "text_heatmap": f"{root}_text_heatmap{ext}",
        "original_box": f"{root}_original_box{ext}",
        "plain_text": f"{root}_plain_text{ext}",
    }


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


def scale_bbox_xywh(bb, original_size, target_size):
    orig_w, orig_h = original_size
    target_w, target_h = target_size
    wr = target_w / orig_w
    hr = target_h / orig_h
    return [(x * wr, y * hr, w * wr, h * hr) for x, y, w, h in bb]


def prepare_clip_image(image):
    if IMAGE_PREPROCESS_MODE == "center_crop":
        return image

    if IMAGE_PREPROCESS_MODE != "resize_stretch":
        raise ValueError(f"Unsupported IMAGE_PREPROCESS_MODE: {IMAGE_PREPROCESS_MODE}")
    return image.resize((CLIP_INPUT_SIZE, CLIP_INPUT_SIZE), Image.BICUBIC)


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



def plot(tokenizer,processor, image_path, text, vmap, tmap, bb=None, save_path=None, save_separate=True):
    
    image = Image.open(image_path).convert('RGB')
    image_for_clip = prepare_clip_image(image)
    
    text_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)]).to(device)
    text_words = tokenizer.convert_ids_to_tokens(text_ids[0].tolist())
    image_under = processor(images=image_for_clip, return_tensors="pt", do_normalize=False)['pixel_values'][0].permute(1,2,0) # no normalization
    original_bb = bb
    scaled_bb = bb
    if bb:
        scaled_bb = scale_bbox_xywh(
            bb=bb,
            original_size=image.size,
            target_size=(image_under.shape[1], image_under.shape[0]),
        )
    if save_path:
        ensure_parent_dir(save_path)
    visualize_vandt_heatmap(tmap, vmap, text_words, image_under, bb=scaled_bb, title=save_path)

    
    if save_path and save_separate:
        text_scores = align_text_scores(tmap, text_words)
        save_separate_figures(
            save_path=save_path,
            image=image_under,
            vmap=vmap,
            text_words=text_words,
            text_scores=text_scores,
            bb=scaled_bb,
        )
    



def generate_plot(img_path,text,method, save_path=None, bb=None, save_separate=True):
    image = Image.open(img_path).convert('RGB')
    image_for_clip = prepare_clip_image(image)
    image_feat = processor(images=image_for_clip, return_tensors="pt")['pixel_values'].to(device)
    image_features = model.get_image_features(image_feat)
    text_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)]).to(device)
    text_words = tokenizer.convert_ids_to_tokens(text_ids[0].tolist())
    text_features = model.get_text_features(text_ids)

    if method == gradcam:
        vmap, tmap = gradcam(model, processor, [text], [image_for_clip], target_layer=GRADCAM_TARGET_LAYER)
    elif method == fast_ig:
        vmap, tmap = fast_ig(model, processor, [text], [image_for_clip], target_layer=ATTENTION_TARGET_LAYER)
    elif method == saliencymap:
        vmap, tmap = saliencymap(model, processor, [text], [image_for_clip], target_layer=ATTENTION_TARGET_LAYER)
    elif method in [chefer,mfaba]:
        vmap, tmap = method(model, processor, [text], [image_for_clip])
    elif method == rise:
        vmap, tmap = rise(model, image_feat,[text_ids],image_features,text_features)
        tmap = [tmap[0].detach().cpu().numpy()]
    elif method == m2ib:
        vmap, tmap = m2ib(model, [text_ids], image_feat, beta=M2IB_RATE, target_layer=M2IB_LAYER)
    elif method == nib:
        vmap, tmap = nib(model, [text_ids], image_feat, NIB_STEPS, NIB_LAYER)
    if vmap.shape[1] == 3:
        vmap = vmap.mean(1)
    if save_path:
        ensure_parent_dir(save_path)
    plot(
        tokenizer,
        processor,
        img_path,
        text,
        vmap.squeeze(),
        tmap[0],
        bb=bb,
        save_path=save_path,
        save_separate=save_separate,
    )



if __name__ == "__main__":
    
    method = m2ib
    #(nib、m2ib、rise、chefer、mfaba、saliencymap、fast_ig、gradcam)
    
    img_path = ""
    text = ""
    bb = None
    save_path = build_save_path(img_path, method)

    print_run_config(
        img_path=img_path,
        text=text,
        method=method,
        save_path=save_path,
        bb=bb,
        save_separate=SAVE_SEPARATE,
    )
    generate_plot(
        img_path,
        text,
        method,
        save_path=save_path,
        bb=bb,
        save_separate=SAVE_SEPARATE,
    )
