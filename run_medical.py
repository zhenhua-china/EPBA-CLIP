import os
import sys
import types
import warnings

warnings.filterwarnings("ignore")


_pyopenssl_stub = types.ModuleType("urllib3.contrib.pyopenssl")
_pyopenssl_stub.inject_into_urllib3 = lambda: None
_pyopenssl_stub.extract_from_urllib3 = lambda: None
sys.modules.setdefault("urllib3.contrib.pyopenssl", _pyopenssl_stub)

import clip
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from clip.model import build_model

from salicncy import chefer, fast_ig, gradcam, m2ib, mfaba, nib, rise, saliencymap
from scripts.clip_wrapper import ClipWrapper


os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CLIP_IMAGE_SIZE = 224
CLIP_MEAN = torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(3, 1, 1)
CLIP_STD = torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(3, 1, 1)


def preprocess(image):
    image = image.convert("RGB")
    width, height = image.size
    if width < height:
        new_width = CLIP_IMAGE_SIZE
        new_height = int(round(height * CLIP_IMAGE_SIZE / width))
    else:
        new_height = CLIP_IMAGE_SIZE
        new_width = int(round(width * CLIP_IMAGE_SIZE / height))

    image = image.resize((new_width, new_height), Image.BICUBIC)
    left = (new_width - CLIP_IMAGE_SIZE) // 2
    top = (new_height - CLIP_IMAGE_SIZE) // 2
    image = image.crop((left, top, left + CLIP_IMAGE_SIZE, top + CLIP_IMAGE_SIZE))

    image_arr = np.asarray(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_arr).permute(2, 0, 1)
    return (image_tensor - CLIP_MEAN) / CLIP_STD



ckpt_path = "clip-imp-pretrained_128_6_after_4.pt"
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(
        f"Medical CLIP checkpoint not found: {ckpt_path}. "
        "Place the checkpoint in the project root before running this script."
    )
state_dict = torch.load(ckpt_path, map_location=device)
if isinstance(state_dict, dict) and "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]
state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

raw_model = build_model(state_dict).to(device)
raw_model.float()
raw_model.eval()


med_model = ClipWrapper(raw_model)
med_model.eval()



def normalize_np(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.min()
    denom = x.max()
    if denom < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return x / denom


def normalize_tensor(x):
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.min()
    denom = x.max()
    if float(denom.detach().cpu()) < 1e-8:
        return torch.zeros_like(x)
    return x / denom


def cosine_score(image_features, text_features):
    return F.cosine_similarity(image_features, text_features, dim=-1).mean()


def tokenize_text(text):
    return clip.tokenize([text]).to(device)


def get_text_words(text):
    
    try:
        from clip.simple_tokenizer import SimpleTokenizer

        tokenizer = SimpleTokenizer()
        token_ids = tokenize_text(text)[0].detach().cpu().tolist()
        eot = int(token_ids.index(49407)) if 49407 in token_ids else len(token_ids) - 1

        words = ["<start>"]
        for token_id in token_ids[1:eot]:
            token = tokenizer.decoder.get(int(token_id), str(token_id))
            token = token.replace("</w>", "")
            words.append(token)
        words.append("<end>")
        return words
    except Exception:
        return ["<start>"] + text.split() + ["<end>"]


def resize_image_for_vis(image_path, image_size=224):
    image = Image.open(image_path).convert("RGB")
    image_vis = image.resize((image_size, image_size), Image.BICUBIC)
    image_under = np.asarray(image_vis).astype(np.float32) / 255.0
    return image, image_under


def scale_bbox_xywh(bb, original_size, target_size=(224, 224)):
    orig_w, orig_h = original_size
    target_w, target_h = target_size

    wr = target_w / orig_w
    hr = target_h / orig_h

    return [(x * wr, y * hr, w * wr, h * hr) for x, y, w, h in bb]


def overlay_heatmap(image, vmap):
    vmap = normalize_np(np.squeeze(vmap))
    heat = plt.get_cmap("jet")(vmap)[..., :3]
    return np.clip(0.45 * image + 0.55 * heat, 0.0, 1.0)


def plot_text_heatmap(ax, tokens, scores, max_width=100, max_height=4, fontsize=14):
    scores = normalize_np(scores)
    x, y = 0.0, max_height * 0.7
    space_width = fontsize * 0.25

    for token, score in zip(tokens, scores):
        alpha = 0.15 + 0.75 * float(score)
        color = (0.2, 0.4, 1.0, alpha)
        bbox = dict(facecolor=color, edgecolor=color, boxstyle="round,pad=0.05", alpha=alpha)
        ax.text(x, y, token, fontsize=fontsize, bbox=bbox)

        token_width = fontsize * 0.45 * max(len(token), 1)
        if x + token_width >= max_width:
            x = 0.0
            y -= 1.0
        else:
            x += token_width + space_width

    ax.set_xlim(0, max_width)
    ax.set_ylim(-max_height, max_height)
    ax.axis("off")



def align_text_scores(tmap, text_words):
    text_scores = np.asarray(tmap).squeeze()
    if text_scores.shape[0] > len(text_words):
        text_scores = text_scores[: len(text_words)]
    elif text_scores.shape[0] < len(text_words):
        pad = len(text_words) - text_scores.shape[0]
        text_scores = np.pad(text_scores, (0, pad), mode="constant")
    return text_scores


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


def save_fig(fig, path, tight=True):
    ensure_parent_dir(path)
    if tight:
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    else:
        fig.savefig(path, dpi=300, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def add_bboxes(ax, bb):
    if bb:
        for x, y, w, h in bb:
            rect = mpl.patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="r", facecolor="none")
            ax.add_patch(rect)


def save_image_panel(path, image, bb=None, vmap=None):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(overlay_heatmap(image, vmap) if vmap is not None else image)
    add_bboxes(ax, bb)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save_fig(fig, path, tight=False)


def plot_plain_text(ax, tokens, max_width=100, max_height=4, fontsize=14):
    x, y = 0.0, max_height * 0.7
    space_width = fontsize * 0.25

    for token in tokens:
        ax.text(x, y, token, fontsize=fontsize, color="black")
        token_width = fontsize * 0.45 * max(len(token), 1)
        if x + token_width >= max_width:
            x = 0.0
            y -= 1.0
        else:
            x += token_width + space_width

    ax.set_xlim(0, max_width)
    ax.set_ylim(-max_height, max_height)
    ax.axis("off")


def save_text_panel(path, tokens, scores=None):
    fig, ax = plt.subplots(figsize=(3, 1.4))
    if scores is None:
        plot_plain_text(ax, tokens)
    else:
        plot_text_heatmap(ax, tokens, scores)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save_fig(fig, path, tight=True)


def save_separate_medical_figures(save_path, image, vmap, text_words, text_scores, bb=None):
    paths = make_separate_save_paths(save_path)
    tokens = text_words[1:-1]
    token_scores = text_scores[1:-1]

    save_image_panel(paths["image_heatmap"], image=image, bb=bb, vmap=vmap)
    save_text_panel(paths["text_heatmap"], tokens=tokens, scores=token_scores)
    save_image_panel(paths["original_box"], image=image, bb=bb, vmap=None)
    save_text_panel(paths["plain_text"], tokens=tokens, scores=None)



def visualize_medical_heatmap(tmap, vmap, text_words, image, title=None, bb=None):
    text_scores = align_text_scores(tmap, text_words)

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(6, 3)

    axs[0].imshow(overlay_heatmap(image, vmap))
    add_bboxes(axs[0], bb)
    axs[0].axis("off")

    plot_text_heatmap(axs[1], text_words[1:-1], text_scores[1:-1])
    plt.tight_layout()

    if title:
        plt.savefig(title, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)



def encode_text_from_embeddings(model, token_embeddings, text_ids):
    x = token_embeddings.type(model.dtype)
    x = x + model.positional_embedding.type(model.dtype)[: x.shape[1], :]
    x = x.permute(1, 0, 2)
    x = model.transformer(x)
    x = x.permute(1, 0, 2)
    x = model.ln_final(x).type(model.dtype)
    x = x[torch.arange(x.shape[0], device=x.device), text_ids.argmax(dim=-1)] @ model.text_projection
    return x


def visual_hidden_before_layer(model, image_feat, target_layer):
    visual = model.visual
    x = visual.conv1(image_feat.type(model.dtype))
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype)
    cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)

    for idx in range(target_layer):
        x = visual.transformer.resblocks[idx](x)
    return x


def visual_output_from_hidden(model, hidden_states, target_layer):
    visual = model.visual
    x = hidden_states
    for idx in range(target_layer, len(visual.transformer.resblocks)):
        x = visual.transformer.resblocks[idx](x)
    x = x.permute(1, 0, 2)
    x = visual.ln_post(x[:, 0, :])
    if visual.proj is not None:
        x = x @ visual.proj
    return x


def text_hidden_before_layer(model, text_ids, target_layer):
    x = model.token_embedding(text_ids).type(model.dtype)
    x = x + model.positional_embedding.type(model.dtype)[: x.shape[1], :]
    x = x.permute(1, 0, 2)

    for idx in range(target_layer):
        x = model.transformer.resblocks[idx](x)
    return x


def text_output_from_hidden(model, hidden_states, text_ids, target_layer):
    x = hidden_states
    for idx in range(target_layer, len(model.transformer.resblocks)):
        x = model.transformer.resblocks[idx](x)
    x = x.permute(1, 0, 2)
    x = model.ln_final(x).type(model.dtype)
    x = x[torch.arange(x.shape[0], device=x.device), text_ids.argmax(dim=-1)] @ model.text_projection
    return x



def medical_gradcam(model, image_feat, text_ids, target_layer=9):
    vision_acts = {}
    vision_grads = {}
    text_acts = {}
    text_grads = {}

    def save_v_act(_, __, output):
        vision_acts["value"] = output

    def save_v_grad(_, __, grad_output):
        vision_grads["value"] = grad_output[0]

    def save_t_act(_, __, output):
        text_acts["value"] = output

    def save_t_grad(_, __, grad_output):
        text_grads["value"] = grad_output[0]

    v_layer = model.visual.transformer.resblocks[target_layer]
    t_layer = model.transformer.resblocks[target_layer]

    v_hook = v_layer.register_forward_hook(save_v_act)
    t_hook = t_layer.register_forward_hook(save_t_act)
    try:
        v_grad_hook = v_layer.register_full_backward_hook(save_v_grad)
        t_grad_hook = t_layer.register_full_backward_hook(save_t_grad)
    except AttributeError:
        v_grad_hook = v_layer.register_backward_hook(save_v_grad)
        t_grad_hook = t_layer.register_backward_hook(save_t_grad)

    try:
        model.zero_grad(set_to_none=True)
        image_features = model.encode_image(image_feat)
        text_features = model.encode_text(text_ids)
        score = cosine_score(image_features, text_features)
        score.backward()

        v_sal = (vision_acts["value"] * vision_grads["value"]).sum(dim=-1).clamp(min=0)
        v_sal = v_sal[1:, 0]
        dim = int(v_sal.numel() ** 0.5)
        vmap = F.interpolate(v_sal.reshape(1, 1, dim, dim), size=224, mode="bilinear", align_corners=False)
        vmap = normalize_tensor(vmap[0, 0]).detach().cpu().numpy()

        t_sal = (text_acts["value"] * text_grads["value"]).sum(dim=-1).clamp(min=0)
        tmap = normalize_tensor(t_sal[:, 0]).detach().cpu().numpy()
    finally:
        v_hook.remove()
        t_hook.remove()
        v_grad_hook.remove()
        t_grad_hook.remove()

    return np.expand_dims(vmap, axis=0), [tmap]


def medical_saliencymap(model, image_feat, text_ids):
    model.zero_grad(set_to_none=True)

    image_var = image_feat.detach().clone().requires_grad_(True)
    text_emb = model.token_embedding(text_ids).detach().clone().requires_grad_(True)

    image_features = model.encode_image(image_var)
    text_features = encode_text_from_embeddings(model, text_emb, text_ids)
    score = cosine_score(image_features, text_features)

    image_grad, text_grad = torch.autograd.grad(score, [image_var, text_emb])

    vmap = image_grad.abs().sum(dim=1)[0]
    tmap = text_grad.abs().sum(dim=-1)[0]

    return (
        np.expand_dims(normalize_tensor(vmap).detach().cpu().numpy(), axis=0),
        [normalize_tensor(tmap).detach().cpu().numpy()],
    )


def medical_fast_ig(model, image_feat, text_ids, steps=20):
    image_base = torch.zeros_like(image_feat)
    text_emb = model.token_embedding(text_ids).detach()
    text_base = torch.zeros_like(text_emb)

    image_grads = torch.zeros_like(image_feat)
    text_grads = torch.zeros_like(text_emb)

    for step in range(1, steps + 1):
        alpha = float(step) / float(steps)

        image_var = (image_base + alpha * (image_feat - image_base)).detach().requires_grad_(True)
        text_var = (text_base + alpha * (text_emb - text_base)).detach().requires_grad_(True)

        image_features = model.encode_image(image_var)
        text_features = encode_text_from_embeddings(model, text_var, text_ids)
        score = cosine_score(image_features, text_features)

        image_grad, text_grad = torch.autograd.grad(score, [image_var, text_var])
        image_grads = image_grads + image_grad.detach()
        text_grads = text_grads + text_grad.detach()

    image_attr = (image_feat - image_base) * image_grads / steps
    text_attr = (text_emb - text_base) * text_grads / steps

    vmap = image_attr.abs().sum(dim=1)[0]
    tmap = text_attr.abs().sum(dim=-1)[0]

    return (
        np.expand_dims(normalize_tensor(vmap).detach().cpu().numpy(), axis=0),
        [normalize_tensor(tmap).detach().cpu().numpy()],
    )


def medical_mfaba(model, image_feat, text_ids, steps=10, step_size=0.01):
    text_features = model.encode_text(text_ids).detach()

    image_hats = [image_feat.detach()]
    image_grads = []
    current = image_feat.detach()
    for _ in range(steps):
        image_var = current.detach().requires_grad_(True)
        score = cosine_score(model.encode_image(image_var), text_features)
        grad = torch.autograd.grad(score, image_var)[0]
        current = (image_var - step_size * grad.sign()).detach()
        image_hats.append(current)
        image_grads.append(grad.detach())

    image_hats = torch.stack(image_hats)
    image_grads = torch.stack(image_grads)
    image_attr = -torch.sum((image_hats[1:] - image_hats[:-1]) * image_grads, dim=0)
    vmap = image_attr.sum(dim=1)[0]

    image_features = model.encode_image(image_feat).detach()
    base_emb = model.token_embedding(text_ids).detach()
    text_hats = [base_emb]
    text_grads = []
    current_emb = base_emb
    for _ in range(steps):
        emb_var = current_emb.detach().requires_grad_(True)
        score = cosine_score(image_features, encode_text_from_embeddings(model, emb_var, text_ids))
        grad = torch.autograd.grad(score, emb_var)[0]
        current_emb = (emb_var - step_size * grad.sign()).detach()
        text_hats.append(current_emb)
        text_grads.append(grad.detach())

    text_hats = torch.stack(text_hats)
    text_grads = torch.stack(text_grads)
    text_attr = -torch.sum((text_hats[1:] - text_hats[:-1]) * text_grads, dim=0)
    tmap = text_attr.sum(dim=-1)[0]

    return (
        np.expand_dims(normalize_tensor(vmap).detach().cpu().numpy(), axis=0),
        [normalize_tensor(tmap).detach().cpu().numpy()],
    )


def medical_nib(model, image_feat, text_ids, num_steps=10, target_layer=9):
    text_features = model.encode_text(text_ids).detach()
    hs_v = visual_hidden_before_layer(model, image_feat, target_layer).detach()

    attribution_v = torch.zeros_like(hs_v)
    for step in range(1, num_steps + 1):
        scaled_hs = (hs_v * float(step) / float(num_steps)).detach().requires_grad_(True)
        image_features = visual_output_from_hidden(model, scaled_hs, target_layer)
        score = cosine_score(image_features, text_features)
        grad = torch.autograd.grad(score, scaled_hs)[0]
        attribution_v = attribution_v + hs_v * grad / num_steps

    v_sal = attribution_v.sum(dim=-1)[1:, 0]
    dim = int(v_sal.numel() ** 0.5)
    vmap = F.interpolate(v_sal.reshape(1, 1, dim, dim), size=224, mode="bilinear", align_corners=False)
    vmap = normalize_tensor(vmap[0, 0]).detach().cpu().numpy()

    image_features = model.encode_image(image_feat).detach()
    hs_t = text_hidden_before_layer(model, text_ids, target_layer).detach()

    attribution_t = torch.zeros_like(hs_t)
    for step in range(1, num_steps + 1):
        scaled_hs = (hs_t * float(step) / float(num_steps)).detach().requires_grad_(True)
        text_features = text_output_from_hidden(model, scaled_hs, text_ids, target_layer)
        score = cosine_score(image_features, text_features)
        grad = torch.autograd.grad(score, scaled_hs)[0]
        attribution_t = attribution_t + hs_t * grad / num_steps

    tmap = normalize_tensor(attribution_t.sum(dim=-1)[:, 0]).detach().cpu().numpy()

    return np.expand_dims(vmap, axis=0), [tmap]


def generate_rise_masks(num_masks=2000, size=224, grid=8, p1=0.1):
    masks = (torch.rand(num_masks, 1, grid, grid, device=device) < p1).float()
    masks = F.interpolate(masks, size=(size, size), mode="bilinear", align_corners=False)
    return masks, p1


def medical_rise(model, image_feat, text_ids, num_masks=2000, gpu_batch=64, p1=0.1):
    text_features = model.encode_text(text_ids).detach()
    masks, p1 = generate_rise_masks(num_masks=num_masks, p1=p1)

    scores = []
    for start in range(0, num_masks, gpu_batch):
        batch_masks = masks[start : start + gpu_batch]
        masked_images = image_feat * batch_masks
        image_features = model.encode_image(masked_images)
        text_batch = text_features.expand(image_features.shape[0], -1)
        scores.append(F.cosine_similarity(image_features, text_batch, dim=-1).detach())

    scores = torch.cat(scores)
    vmap = (scores[:, None, None] * masks[:, 0]).mean(dim=0) / p1

    base_emb = model.token_embedding(text_ids).detach()
    seq_len = base_emb.shape[1]
    token_masks = (torch.rand(num_masks, seq_len, 1, device=device) < p1).float()
    token_masks[:, 0, :] = 1.0
    token_masks[:, int(text_ids.argmax(dim=-1).item()), :] = 1.0

    image_features = model.encode_image(image_feat).detach()
    text_scores = []
    for start in range(0, num_masks, gpu_batch):
        batch_masks = token_masks[start : start + gpu_batch]
        masked_emb = base_emb.expand(batch_masks.shape[0], -1, -1) * batch_masks
        text_features = encode_text_from_embeddings(model, masked_emb, text_ids.expand(batch_masks.shape[0], -1))
        image_batch = image_features.expand(text_features.shape[0], -1)
        text_scores.append(F.cosine_similarity(image_batch, text_features, dim=-1).detach())

    text_scores = torch.cat(text_scores)
    tmap = (text_scores[:, None] * token_masks.squeeze(-1)).mean(dim=0) / p1

    return (
        np.expand_dims(normalize_tensor(vmap).detach().cpu().numpy(), axis=0),
        [normalize_tensor(tmap).detach().cpu().numpy()],
    )


def run_medical_method(method, image_feat, text_ids):
    name = getattr(method, "__name__", str(method))

    if name == "m2ib":
        return m2ib(med_model, [text_ids], image_feat, 0.1)

    if name == "nib":
        return medical_nib(raw_model, image_feat, text_ids, num_steps=10, target_layer=9)

    if name == "rise":
        return medical_rise(raw_model, image_feat, text_ids)

    if name in ("chefer", "gradcam"):
        return medical_gradcam(raw_model, image_feat, text_ids, target_layer=9)

    if name == "fast_ig":
        return medical_fast_ig(raw_model, image_feat, text_ids, steps=20)

    if name == "mfaba":
        return medical_mfaba(raw_model, image_feat, text_ids, steps=10)

    if name == "saliencymap":
        return medical_saliencymap(raw_model, image_feat, text_ids)

    raise ValueError(f"Unsupported medical explanation method: {name}")



def plot_medical_result(image_path, text, vmap, tmap, bb=None, save_path=None, save_separate=True):
    image, image_under = resize_image_for_vis(image_path, image_size=224)
    text_words = get_text_words(text)

    if bb is not None:
        bb = scale_bbox_xywh(bb=bb, original_size=image.size, target_size=(224, 224))

    visualize_medical_heatmap(
        tmap=tmap,
        vmap=vmap,
        text_words=text_words,
        image=image_under,
        bb=bb,
        title=save_path,
    )

    
    if save_path is not None and save_separate:
        text_scores = align_text_scores(tmap, text_words)
        save_separate_medical_figures(
            save_path=save_path,
            image=image_under,
            vmap=vmap,
            text_words=text_words,
            text_scores=text_scores,
            bb=bb,
        )
  


def generate_medical_plot(img_path, text, method=m2ib, save_path=None, bb=None, save_separate=True):
    image = Image.open(img_path).convert("RGB")
    image_feat = preprocess(image).unsqueeze(0).to(device)
    text_ids = tokenize_text(text)

    vmap, tmap = run_medical_method(method, image_feat, text_ids)

    vmap = np.asarray(vmap).squeeze()
    if vmap.ndim == 3 and vmap.shape[0] == 3:
        vmap = vmap.mean(axis=0)

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    plot_medical_result(
        image_path=img_path,
        text=text,
        vmap=vmap,
        tmap=tmap[0],
        bb=bb,
        save_path=save_path,
        save_separate=save_separate,
    )



if __name__ == "__main__":
    
    img_path = ""
    text = ""
    save_path = ""

    # xywh format: x, y, width, height
    bb = [
        (347, 605, 600, 1052),
        (1337, 765, 899, 1127),
    ]

    if not os.path.exists(img_path):
        raise FileNotFoundError(
            f"Medical example image not found: {img_path}. "
            "Please provide a permitted local image before running this script."
        )

    generate_medical_plot(
        img_path=img_path,
        text=text,
        method=m2ib,
        save_path=save_path,
        bb=bb,
    )
