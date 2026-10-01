import os
import sys
import types
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

_pyopenssl_stub = types.ModuleType("urllib3.contrib.pyopenssl")
_pyopenssl_stub.inject_into_urllib3 = lambda: None
_pyopenssl_stub.extract_from_urllib3 = lambda: None
sys.modules.setdefault("urllib3.contrib.pyopenssl", _pyopenssl_stub)

import clip
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from clip.model import build_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CLIP_IMAGE_SIZE = 224
DEFAULT_MEDICAL_LAYERS = tuple(range(12))
DEFAULT_TOP_K = 5
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
            words.append(token.replace("</w>", ""))
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


def save_fig(fig, path, tight=True, dpi=300):
    ensure_parent_dir(path)
    if tight:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    else:
        fig.savefig(path, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def add_bboxes(ax, bb, linewidth=2):
    if bb:
        for x, y, w, h in bb:
            rect = mpl.patches.Rectangle((x, y), w, h, linewidth=linewidth, edgecolor="r", facecolor="none")
            ax.add_patch(rect)


def image_to_float_array(image):
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    image = np.asarray(image).astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return image


def resize_vmap_to_image(vmap, image):
    image_w, image_h = image.size if isinstance(image, Image.Image) else (image.shape[1], image.shape[0])
    vmap = normalize_np(np.squeeze(vmap))
    vmap_img = Image.fromarray((vmap * 255).astype(np.uint8))
    vmap_img = vmap_img.resize((image_w, image_h), Image.BICUBIC)
    return np.asarray(vmap_img).astype(np.float32) / 255.0


def save_image_panel(path, image, bb=None, vmap=None):
    image_arr = image_to_float_array(image)
    image_h, image_w = image_arr.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(figsize=(image_w / dpi, image_h / dpi), dpi=dpi)
    if vmap is not None and np.squeeze(vmap).shape != image_arr.shape[:2]:
        vmap = resize_vmap_to_image(vmap, image_arr)
    ax.imshow(overlay_heatmap(image_arr, vmap) if vmap is not None else image_arr)
    add_bboxes(ax, bb, linewidth=max(2.0, min(image_w, image_h) / 280.0))
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save_fig(fig, path, tight=False, dpi=dpi)


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
        ensure_parent_dir(title)
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


def capture_layer_output(model, image_feat, text_ids, tower, layer_idx):
    outputs = {}
    if tower == "vision":
        layer = model.visual.transformer.resblocks[layer_idx]

        def hook(_, __, output):
            outputs["value"] = output.detach()

        handle = layer.register_forward_hook(hook)
        try:
            with torch.no_grad():
                model.encode_image(image_feat)
        finally:
            handle.remove()
    else:
        layer = model.transformer.resblocks[layer_idx]

        def hook(_, __, output):
            outputs["value"] = output.detach()

        handle = layer.register_forward_hook(hook)
        try:
            with torch.no_grad():
                model.encode_text(text_ids)
        finally:
            handle.remove()

    return outputs["value"]


def get_resblock(model, tower, layer_idx):
    if tower == "vision":
        return model.visual.transformer.resblocks[layer_idx]
    return model.transformer.resblocks[layer_idx]


def set_resblock(model, tower, layer_idx, layer):
    if tower == "vision":
        model.visual.transformer.resblocks[layer_idx] = layer
    else:
        model.transformer.resblocks[layer_idx] = layer



class ResidualAwareBottleneckOpenAI(nn.Module):
    def __init__(
        self,
        features,
        beta=0.1,
        sigma_fixed=1.0,
        gamma_init=0.5,
        learnable_gamma=True,
        residual_mode="centered",
        eps=1e-6,
    ):
        super().__init__()
        features = features.detach().float()
        if features.dim() != 3:
            raise ValueError(f"Expected [seq,batch,width] features, got {features.shape}")

        mean = features.mean(dim=1)
        std = torch.ones_like(mean) * float(sigma_fixed)

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.alpha = nn.Parameter(torch.full_like(mean, fill_value=5.0))
        self.beta = beta
        self.sigma_fixed = sigma_fixed
        self.residual_mode = residual_mode
        self.eps = eps

        gamma_init = float(np.clip(gamma_init, 1e-4, 1.0 - 1e-4))
        gamma_logit = torch.logit(torch.tensor(gamma_init, dtype=torch.float32, device=features.device))
        if learnable_gamma:
            self.gamma_logit = nn.Parameter(gamma_logit)
        else:
            self.register_buffer("gamma_logit", gamma_logit)

        self.buffer_capacity = None
        self.buffer_gamma = None

    @staticmethod
    def calc_capacity(mu, var):
        var = torch.clamp(var, min=1e-8)
        return -0.5 * (1 + torch.log(var) - mu**2 - var)

    def reset_alpha(self):
        with torch.no_grad():
            self.alpha.fill_(5.0)

    def residual_feature(self, x, layer_mean):
        if self.residual_mode == "centered":
            return x.detach() - layer_mean
        if self.residual_mode == "detached":
            return x.detach()
        if self.residual_mode == "token_centered":
            return x.detach() - x.detach().mean(dim=0, keepdim=True)
        if self.residual_mode == "channel_centered":
            return x.detach() - x.detach().mean(dim=-1, keepdim=True)
        if self.residual_mode == "zero":
            return torch.zeros_like(x)
        raise ValueError(f"Unknown residual_mode: {self.residual_mode}")

    def forward(self, x):
        h = torch.sigmoid(self.alpha).unsqueeze(1).expand_as(x)
        layer_mean = self.mean.unsqueeze(1).expand_as(x)
        gamma = torch.sigmoid(self.gamma_logit)

        v_res = self.residual_feature(x, layer_mean)
        replacement_mean = gamma * v_res + (1.0 - gamma) * layer_mean
        sigma = torch.full_like(x, fill_value=float(self.sigma_fixed))

        z_mean = h * x + (1.0 - h) * replacement_mean
        z_std = (1.0 - h) * sigma
        z_var = torch.clamp(z_std**2, min=self.eps)
        z = z_mean + z_std * torch.randn_like(x)

        self.buffer_capacity = torch.nan_to_num(
            self.calc_capacity(z_mean, z_var),
            nan=0.0,
            posinf=1e4,
            neginf=0.0,
        )
        self.buffer_gamma = gamma.detach()
        return z


def pair_cosine_similarity_openai(model, text_ids, image_feat):
    model.eval()
    with torch.no_grad():
        text_feat = model.encode_text(text_ids)
        image_feat_out = model.encode_image(image_feat)
        return float(cosine_score(image_feat_out, text_feat).detach().cpu().item())


def train_bottleneck(model, bottleneck, text_ids, image_feat, batch_size=10, lr=0.01, train_steps=10):
    optimizer = torch.optim.Adam(bottleneck.parameters(), lr=lr)
    text_batch = text_ids.expand(batch_size, -1)
    image_batch = image_feat.expand(batch_size, -1, -1, -1)
    fitting_estimator = torch.nn.CosineSimilarity(eps=1e-6)

    bottleneck.reset_alpha()
    model.eval()

    for _ in range(train_steps):
        optimizer.zero_grad()
        text_out = model.encode_text(text_batch)
        image_out = model.encode_image(image_batch)
        compression_term = torch.nan_to_num(bottleneck.buffer_capacity.mean(), nan=0.0, posinf=1e4, neginf=0.0)
        fitting_term = fitting_estimator(text_out, image_out).mean()
        total = bottleneck.beta * compression_term - fitting_term
        if not torch.isfinite(total):
            break
        total.backward()
        torch.nn.utils.clip_grad_norm_(bottleneck.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            bottleneck.alpha.clamp_(-8.0, 8.0)
            bottleneck.gamma_logit.clamp_(-8.0, 8.0)

    return compression_term, fitting_term, total


def safe_float(value, default=0.0):
    if torch.is_tensor(value):
        value = value.detach().cpu().item()
    value = float(value)
    return value if np.isfinite(value) else default


def run_single_layer_medical_eviba(
    model,
    image_feat,
    text_ids,
    tower,
    layer_idx,
    beta=0.1,
    var=1.0,
    lr=0.01,
    train_steps=10,
    sigma_fixed=0.1,
    gamma_init=0.5,
    learnable_gamma=True,
    residual_mode="centered",
):
    features = capture_layer_output(model, image_feat, text_ids, tower=tower, layer_idx=layer_idx)
    bottleneck = ResidualAwareBottleneckOpenAI(
        features=features,
        beta=beta,
        sigma_fixed=sigma_fixed if sigma_fixed is not None else var,
        gamma_init=gamma_init,
        learnable_gamma=learnable_gamma,
        residual_mode=residual_mode,
    ).to(device)

    original_layer = get_resblock(model, tower, layer_idx)
    replacement = nn.Sequential(original_layer, bottleneck)

    original_sim = pair_cosine_similarity_openai(model, text_ids, image_feat)
    set_resblock(model, tower, layer_idx, replacement)
    try:
        loss_c, loss_f, loss_t = train_bottleneck(
            model=model,
            bottleneck=bottleneck,
            text_ids=text_ids,
            image_feat=image_feat,
            lr=lr,
            train_steps=train_steps,
        )
        capacity = bottleneck.buffer_capacity.detach().mean(dim=1)
    finally:
        set_resblock(model, tower, layer_idx, original_layer)

    final_sim = safe_float(loss_f)
    kl = safe_float(loss_c)
    loss = safe_float(loss_t)
    gamma = safe_float(bottleneck.buffer_gamma, default=0.5)

    info = {
        "original_sim": original_sim,
        "final_sim": final_sim,
        "sim_drop": max(0.0, original_sim - final_sim),
        "kl": kl,
        "loss": loss,
        "gamma": gamma,
        "layer": layer_idx,
    }

    if tower == "vision":
        saliency = capacity.sum(dim=-1)[1:]
        dim = int(saliency.numel() ** 0.5)
        saliency = saliency.reshape(1, 1, dim, dim)
        saliency = F.interpolate(saliency, size=224, mode="bilinear", align_corners=False)
        return normalize_tensor(saliency[0, 0]).detach().cpu().numpy(), info

    saliency = capacity.sum(dim=-1)
    return normalize_tensor(saliency).detach().cpu().numpy(), info


def compute_layer_weights(layer_infos, top_k=DEFAULT_TOP_K, kl_tau=1.0, eps=1e-6):
    scores = []
    for info in layer_infos:
        sim_drop = max(0.0, float(info["sim_drop"]))
        kl = max(float(info["kl"]), eps)
        score = sim_drop / ((kl + eps) ** kl_tau)
        scores.append(score if np.isfinite(score) else 0.0)

    scores = np.asarray(scores, dtype=np.float64)
    if np.sum(scores) < eps:
        weights = np.zeros_like(scores)
        weights[-1] = 1.0
        return weights

    if top_k is not None and top_k < len(scores):
        keep_idx = np.argsort(scores)[-top_k:]
        mask = np.zeros_like(scores)
        mask[keep_idx] = 1.0
        scores = scores * mask

    return scores / (np.sum(scores) + eps)


def fuse_maps(maps, weights):
    fused = None
    for saliency_map, weight in zip(maps, weights):
        saliency_map = normalize_np(saliency_map)
        fused = weight * saliency_map if fused is None else fused + weight * saliency_map
    return normalize_np(fused)


def run_tower_multilayer(
    model,
    image_feat,
    text_ids,
    tower,
    layers=DEFAULT_MEDICAL_LAYERS,
    beta=0.1,
    var=1.0,
    lr=0.01,
    train_steps=10,
    sigma_fixed=0.1,
    gamma_init=0.5,
    learnable_gamma=True,
    residual_mode="centered",
    top_k=DEFAULT_TOP_K,
    kl_tau=1.0,
):
    maps = []
    infos = []
    for layer_idx in layers:
        saliency_map, info = run_single_layer_medical_eviba(
            model=model,
            image_feat=image_feat,
            text_ids=text_ids,
            tower=tower,
            layer_idx=layer_idx,
            beta=beta,
            var=var,
            lr=lr,
            train_steps=train_steps,
            sigma_fixed=sigma_fixed,
            gamma_init=gamma_init,
            learnable_gamma=learnable_gamma,
            residual_mode=residual_mode,
        )
        maps.append(saliency_map)
        infos.append(info)

    weights = compute_layer_weights(infos, top_k=top_k, kl_tau=kl_tau)
    return fuse_maps(maps, weights), {"layers": list(layers), "weights": weights, "infos": infos, "maps": maps}


def medical_eviba(
    model,
    image_feat,
    text_ids,
    layers=DEFAULT_MEDICAL_LAYERS,
    beta=0.1,
    var=1.0,
    lr=0.01,
    train_steps=10,
    sigma_fixed=0.1,
    gamma_init=0.5,
    learnable_gamma=True,
    residual_mode="centered",
    top_k=DEFAULT_TOP_K,
    kl_tau=1.0,
    return_details=True,
):
    vmap, vdetail = run_tower_multilayer(
        model=model,
        image_feat=image_feat,
        text_ids=text_ids,
        tower="vision",
        layers=layers,
        beta=beta,
        var=var,
        lr=lr,
        train_steps=train_steps,
        sigma_fixed=sigma_fixed,
        gamma_init=gamma_init,
        learnable_gamma=learnable_gamma,
        residual_mode=residual_mode,
        top_k=top_k,
        kl_tau=kl_tau,
    )
    tmap, tdetail = run_tower_multilayer(
        model=model,
        image_feat=image_feat,
        text_ids=text_ids,
        tower="text",
        layers=layers,
        beta=beta,
        var=var,
        lr=lr,
        train_steps=train_steps,
        sigma_fixed=sigma_fixed,
        gamma_init=gamma_init,
        learnable_gamma=learnable_gamma,
        residual_mode=residual_mode,
        top_k=top_k,
        kl_tau=kl_tau,
    )

    if return_details:
        return np.expand_dims(vmap, axis=0), [tmap], {"vision": [vdetail], "text": [tdetail]}
    return np.expand_dims(vmap, axis=0), [tmap]



def plot_medical_eviba_result(image_path, text, vmap, tmap, bb=None, save_path=None, save_separate=True):
    image, image_under = resize_image_for_vis(image_path, image_size=224)
    text_words = get_text_words(text)

    scaled_bb = bb
    if bb is not None:
        scaled_bb = scale_bbox_xywh(bb=bb, original_size=image.size, target_size=(224, 224))

    visualize_medical_heatmap(
        tmap=tmap,
        vmap=vmap,
        text_words=text_words,
        image=image_under,
        bb=scaled_bb,
        title=save_path,
    )

    if save_path is not None and save_separate:
        text_scores = align_text_scores(tmap, text_words)
        save_separate_medical_figures(
            save_path=save_path,
            image=image,
            vmap=vmap,
            text_words=text_words,
            text_scores=text_scores,
            bb=bb,
        )


def generate_medical_eviba_plot(img_path, text, save_path=None, bb=None, save_separate=True):
    image = Image.open(img_path).convert("RGB")
    image_feat = preprocess(image).unsqueeze(0).to(device)
    text_ids = tokenize_text(text)

    vmap, tmap, details = medical_eviba(
        model=raw_model,
        image_feat=image_feat,
        text_ids=text_ids,
        layers=DEFAULT_MEDICAL_LAYERS,
        beta=0.1,
        var=1.0,
        lr=0.01,
        train_steps=10,
        sigma_fixed=0.1,
        gamma_init=0.5,
        learnable_gamma=True,
        residual_mode="centered",
        top_k=DEFAULT_TOP_K,
        kl_tau=1.0,
        return_details=True,
    )

    print("Vision layer weights:", details["vision"][0]["weights"])
    print("Vision layer infos:")
    for info in details["vision"][0]["infos"]:
        print(info)

    print("Text layer weights:", details["text"][0]["weights"])
    print("Text layer infos:")
    for info in details["text"][0]["infos"]:
        print(info)

    vmap = np.asarray(vmap).squeeze()
    if vmap.ndim == 3 and vmap.shape[0] == 3:
        vmap = vmap.mean(axis=0)

    if save_path is not None:
        ensure_parent_dir(save_path)

    plot_medical_eviba_result(
        image_path=img_path,
        text=text,
        vmap=vmap,
        tmap=tmap[0],
        bb=bb,
        save_path=save_path,
        save_separate=save_separate,
    )

    return vmap, tmap, details


if __name__ == "__main__":
    
    img_path = ""
    text = ""
    save_path = ""

    # bb = [
    #     (1510.0,1268.0,755.0,796.0),
    #     (454.0,690.0,637.0,670.0),
    # ]

    bb = [
        (1704.0,1669.0,839.0,964.0),
    ]

    if not os.path.exists(img_path):
        raise FileNotFoundError(
            f"Medical example image not found: {img_path}. "
            "Please provide a permitted local image before running this script."
        )

    generate_medical_eviba_plot(
        img_path=img_path,
        text=text,
        save_path=save_path,
        bb=bb,
    )
