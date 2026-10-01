import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.iba import Estimator
from scripts.utils import replace_layer, normalize, mySequential

try:
    from AdaptiveNoiseGenerator import AdaptiveNoiseGenerator
except Exception:
    AdaptiveNoiseGenerator = None




def extract_feature_map(model, layer_idx, x):
    with torch.no_grad():
        states = model(x, output_hidden_states=True)
        feature = states["hidden_states"][layer_idx + 1]
        return feature


def extract_bert_layer(model, layer_idx):
    for _, submodule in model.named_children():
        for n, s in submodule.named_children():
            if n == "layers" or n == "resblocks":
                for n2, s2 in s.named_children():
                    if n2 == str(layer_idx):
                        return s2
    raise RuntimeError(f"Cannot find layer {layer_idx} in model.")


def get_compression_estimator(var, layer, features):
    estimator = Estimator(layer)
    estimator.M = torch.zeros_like(features)
    estimator.S = var * np.ones(features.shape)
    estimator.N = 1
    estimator.layer = layer
    return estimator


def build_noise_generator(features, noise_mode, sigma_fixed, sigma_min, sigma_max):
    if noise_mode is None:
        return None

    if AdaptiveNoiseGenerator is None:
        raise ImportError("AdaptiveNoiseGenerator is not found.")

    return AdaptiveNoiseGenerator(
        seq_len=features.shape[1],
        noise_mode=noise_mode,
        sigma_fixed=sigma_fixed,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        device=features.device,
    )


def pair_cosine_similarity(model, text_t, image_t):
    model.eval()
    with torch.no_grad():
        text_feat = model.get_text_features(text_t)
        image_feat = model.get_image_features(image_t)
        sim = F.cosine_similarity(text_feat, image_feat, dim=-1).mean()
    return sim.item()


def safe_normalize_np(x):
    x = np.asarray(x)
    x = x - np.nanmin(x)
    denom = np.nanmax(x) - np.nanmin(x)
    if denom < 1e-8:
        return np.zeros_like(x)
    return x / denom




class ResidualAwareInformationBottleneck(nn.Module):
    def __init__(
        self,
        mean,
        std,
        device=None,
        noise_generator=None,
        sigma_fixed=1.0,
        gamma_init=0.5,
        learnable_gamma=True,
        residual_mode="centered",
        eps=1e-6,
    ):
        super().__init__()

        self.device = device
        self.initial_value = 5.0
        self.eps = eps
        self.residual_mode = residual_mode
        self.noise_generator = noise_generator
        self.sigma_fixed = sigma_fixed

        self.std = torch.tensor(std, dtype=torch.float, device=self.device, requires_grad=False)
        self.mean = torch.tensor(mean, dtype=torch.float, device=self.device, requires_grad=False)

        self.alpha = nn.Parameter(
            torch.full((1, *self.mean.shape), fill_value=self.initial_value, device=self.device)
        )
        self.sigmoid = nn.Sigmoid()

        gamma_init = float(np.clip(gamma_init, 1e-4, 1.0 - 1e-4))
        gamma_logit = torch.logit(torch.tensor(gamma_init, dtype=torch.float, device=self.device))

        if learnable_gamma:
            self.gamma_logit = nn.Parameter(gamma_logit)
        else:
            self.register_buffer("gamma_logit", gamma_logit)

        self.buffer_capacity = None
        self.buffer_lambda = None
        self.buffer_gamma = None
        self.buffer_z_mean = None
        self.buffer_z_var = None

        self.reset_alpha()

    @staticmethod
    def _calc_capacity(mu, var):
        var = torch.clamp(var, min=1e-8)
        kl = -0.5 * (1 + torch.log(var) - mu ** 2 - var)
        return kl

    def reset_alpha(self):
        with torch.no_grad():
            self.alpha.fill_(self.initial_value)
        return self.alpha

    def _expand_mean_like(self, x):
        mu = self.mean
        while mu.dim() < x.dim():
            mu = mu.unsqueeze(0)
        return mu.expand_as(x)

    def _get_sigma(self, x):
        if self.noise_generator is None:
            sigma = torch.full_like(x, fill_value=self.sigma_fixed)
        else:
            sigma = self.noise_generator(x)
            if sigma.shape[-1] == 1 and x.shape[-1] != 1:
                sigma = sigma.expand(x.shape[0], x.shape[1], x.shape[2])
        return torch.clamp(sigma, min=self.eps)

    def _get_residual_feature(self, x, layer_mean):
        if self.residual_mode == "centered":
            
            return x.detach() - layer_mean

        elif self.residual_mode == "detached":
            
            return x.detach()

        elif self.residual_mode == "token_centered":
            
            return x.detach() - x.detach().mean(dim=1, keepdim=True)

        elif self.residual_mode == "channel_centered":
            
            return x.detach() - x.detach().mean(dim=-1, keepdim=True)

        elif self.residual_mode == "zero":
            return torch.zeros_like(x)

        else:
            raise ValueError(f"Unknown residual_mode: {self.residual_mode}")

    def forward(self, x, **kwargs):
        h = self.sigmoid(self.alpha)
        h = h.expand(x.shape[0], x.shape[1], -1)

        layer_mean = self._expand_mean_like(x)
        gamma = torch.sigmoid(self.gamma_logit)

        sigma = self._get_sigma(x)
        eps = torch.randn_like(x)

        v_res = self._get_residual_feature(x, layer_mean)

        replacement_mean = gamma * v_res + (1.0 - gamma) * layer_mean

        z_mean = h * x + (1.0 - h) * replacement_mean
        z_std = (1.0 - h) * sigma
        z_var = torch.clamp(z_std ** 2, min=self.eps)

        z = z_mean + z_std * eps

        self.buffer_capacity = self._calc_capacity(z_mean, z_var)
        self.buffer_lambda = h.detach()
        self.buffer_gamma = gamma.detach()
        self.buffer_z_mean = z_mean.detach()
        self.buffer_z_var = z_var.detach()

        return (z,)




class ResidualAwareIBAInterpreter:
    def __init__(
        self,
        model,
        estim,
        beta,
        steps=10,
        lr=1,
        batch_size=10,
        progbar=False,
        noise_generator=None,
        sigma_fixed=1.0,
        gamma_init=0.5,
        learnable_gamma=True,
        residual_mode="centered",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.original_layer = estim.get_layer()
        self.shape = estim.shape()
        self.beta = beta
        self.batch_size = batch_size
        self.lr = lr
        self.train_steps = steps
        self.progbar = progbar

        self.fitting_estimator = torch.nn.CosineSimilarity(eps=1e-6)

        self.bottleneck = ResidualAwareInformationBottleneck(
            estim.mean(),
            estim.std(),
            device=self.device,
            noise_generator=noise_generator,
            sigma_fixed=sigma_fixed,
            gamma_init=gamma_init,
            learnable_gamma=learnable_gamma,
            residual_mode=residual_mode,
        )

        self.sequential = mySequential(self.original_layer, self.bottleneck)

    def text_heatmap(self, text_t, image_t, return_info=False):
        saliency, info = self._run_text_training(text_t, image_t)
        saliency = torch.nansum(saliency, -1).cpu().detach().numpy()
        saliency = normalize(saliency)

        if return_info:
            return normalize(saliency), info
        return normalize(saliency)

    def vision_heatmap(self, text_t, image_t, return_info=False):
        saliency, info = self._run_vision_training(text_t, image_t)

        saliency = torch.nansum(saliency, -1)[1:]  # remove CLS token
        dim = int(saliency.numel() ** 0.5)
        saliency = saliency.reshape(1, 1, dim, dim)
        saliency = F.interpolate(saliency, size=224, mode="bilinear", align_corners=False)
        saliency = saliency.squeeze().cpu().detach().numpy()
        saliency = normalize(saliency)

        if return_info:
            return saliency, info
        return saliency

    def _run_text_training(self, text_t, image_t):
        original_sim = pair_cosine_similarity(self.model, text_t, image_t)

        replace_layer(self.model.text_model, self.original_layer, self.sequential)
        loss_c, loss_f, loss_t = self._train_bottleneck(text_t, image_t)
        replace_layer(self.model.text_model, self.sequential, self.original_layer)

        info = self._build_info(original_sim, loss_c, loss_f, loss_t)
        return self.bottleneck.buffer_capacity.mean(axis=0), info

    def _run_vision_training(self, text_t, image_t):
        original_sim = pair_cosine_similarity(self.model, text_t, image_t)

        replace_layer(self.model.vision_model, self.original_layer, self.sequential)
        loss_c, loss_f, loss_t = self._train_bottleneck(text_t, image_t)
        replace_layer(self.model.vision_model, self.sequential, self.original_layer)

        info = self._build_info(original_sim, loss_c, loss_f, loss_t)
        return self.bottleneck.buffer_capacity.mean(axis=0), info

    def _train_bottleneck(self, text_t, image_t):
        batch = (
            text_t.expand(self.batch_size, -1),
            image_t.expand(self.batch_size, -1, -1, -1),
        )

        optimizer = torch.optim.Adam(lr=self.lr, params=self.bottleneck.parameters())

        self.bottleneck.reset_alpha()
        self.model.eval()

        for _ in range(self.train_steps):
            optimizer.zero_grad()

            text_out = self.model.get_text_features(batch[0])
            image_out = self.model.get_image_features(batch[1])

            loss_c, loss_f, loss_t = self.calc_loss(text_out, image_out)

            loss_t.backward()
            optimizer.step()

        return loss_c, loss_f, loss_t

    def calc_loss(self, outputs, labels):
        compression_term = self.bottleneck.buffer_capacity.mean()
        fitting_term = self.fitting_estimator(outputs, labels).mean()

        # minimize: beta * KL - cosine
        total = self.beta * compression_term - fitting_term

        return compression_term, fitting_term, total

    def _build_info(self, original_sim, loss_c, loss_f, loss_t):
        final_sim = float(loss_f.detach().cpu().item())
        kl = float(loss_c.detach().cpu().item())
        sim_drop = max(0.0, original_sim - final_sim)

        return {
            "original_sim": float(original_sim),
            "final_sim": final_sim,
            "sim_drop": sim_drop,
            "kl": kl,
            "loss": float(loss_t.detach().cpu().item()),
            "gamma": float(self.bottleneck.buffer_gamma.detach().cpu().item()),
        }




def vision_heatmap_residual_iba(
    text_t,
    image_t,
    model,
    layer_idx=9,
    beta=0.1,
    var=1,
    lr=1,
    train_steps=10,
    progbar=False,
    noise_mode=None,
    sigma_fixed=1.0,
    sigma_min=0.05,
    sigma_max=1.0,
    gamma_init=0.5,
    learnable_gamma=True,
    residual_mode="centered",
    return_info=False,
):
    features = extract_feature_map(model.vision_model, layer_idx, image_t)
    layer = extract_bert_layer(model.vision_model, layer_idx)
    compression_estimator = get_compression_estimator(var, layer, features)

    noise_generator = build_noise_generator(
        features, noise_mode, sigma_fixed, sigma_min, sigma_max
    )

    reader = ResidualAwareIBAInterpreter(
        model,
        compression_estimator,
        beta=beta,
        lr=lr,
        steps=train_steps,
        progbar=progbar,
        noise_generator=noise_generator,
        sigma_fixed=sigma_fixed,
        gamma_init=gamma_init,
        learnable_gamma=learnable_gamma,
        residual_mode=residual_mode,
    )

    return reader.vision_heatmap(text_t, image_t, return_info=return_info)


def text_heatmap_residual_iba(
    text_t,
    image_t,
    model,
    layer_idx=9,
    beta=0.1,
    var=1,
    lr=1,
    train_steps=10,
    progbar=False,
    noise_mode=None,
    sigma_fixed=1.0,
    sigma_min=0.05,
    sigma_max=1.0,
    gamma_init=0.5,
    learnable_gamma=True,
    residual_mode="centered",
    return_info=False,
):
    features = extract_feature_map(model.text_model, layer_idx, text_t)
    layer = extract_bert_layer(model.text_model, layer_idx)
    compression_estimator = get_compression_estimator(var, layer, features)

    noise_generator = build_noise_generator(
        features, noise_mode, sigma_fixed, sigma_min, sigma_max
    )

    reader = ResidualAwareIBAInterpreter(
        model,
        compression_estimator,
        beta=beta,
        lr=lr,
        steps=train_steps,
        progbar=progbar,
        noise_generator=noise_generator,
        sigma_fixed=sigma_fixed,
        gamma_init=gamma_init,
        learnable_gamma=learnable_gamma,
        residual_mode=residual_mode,
    )

    return reader.text_heatmap(text_t, image_t, return_info=return_info)



def compute_layer_scores(layer_infos, kl_tau=1.0, eps=1e-6):
    scores = []

    for info in layer_infos:
        sim_drop = max(0.0, float(info["sim_drop"]))
        kl = max(float(info["kl"]), eps)
        score = sim_drop / ((kl + eps) ** kl_tau)
        scores.append(score)

    return np.asarray(scores, dtype=np.float64)


def compute_layer_weights(layer_infos, temperature=0.5, kl_tau=1.0, eps=1e-6, top_k=5):
    """
    Contribution-aware Top-K layer fusion.
    Keep the Top-K highest contribution scores and normalize them linearly.
    """
    scores = compute_layer_scores(layer_infos, kl_tau=kl_tau, eps=eps)

    if len(scores) == 0:
        return scores

    
    if np.sum(scores) < eps:
        return np.ones_like(scores) / len(scores)

    
    if top_k is not None and top_k < len(scores):
        keep_idx = np.argsort(scores)[-top_k:]
        mask = np.zeros_like(scores)
        mask[keep_idx] = 1.0
        scores = scores * mask

    weights = scores / (np.sum(scores) + eps)

    return weights


def get_selected_layers(layers, weights, eps=1e-12):
    return [int(layer) for layer, weight in zip(layers, weights) if float(weight) > eps]


def fuse_maps(maps, weights):
    fused = None

    for m, w in zip(maps, weights):
        m = safe_normalize_np(m)
        if fused is None:
            fused = w * m
        else:
            fused = fused + w * m

    return safe_normalize_np(fused)


def vision_heatmap_eviba_multilayer(
    text_t,
    image_t,
    model,
    layers=(7, 8, 9),
    beta=0.1,
    var=1,
    lr=1,
    train_steps=10,
    progbar=False,
    noise_mode=None,
    sigma_fixed=1.0,
    sigma_min=0.05,
    sigma_max=1.0,
    gamma_init=0.5,
    learnable_gamma=True,
    residual_mode="centered",
    temperature=0.5,
    kl_tau=1.0,
    top_k=5,
    return_details=False,
):
    maps = []
    infos = []

    for layer_idx in layers:
        hmap, info = vision_heatmap_residual_iba(
            text_t=text_t,
            image_t=image_t,
            model=model,
            layer_idx=layer_idx,
            beta=beta,
            var=var,
            lr=lr,
            train_steps=train_steps,
            progbar=progbar,
            noise_mode=noise_mode,
            sigma_fixed=sigma_fixed,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            gamma_init=gamma_init,
            learnable_gamma=learnable_gamma,
            residual_mode=residual_mode,
            return_info=True,
        )

        info["layer"] = layer_idx
        maps.append(hmap)
        infos.append(info)

    weights = compute_layer_weights(
        infos,
        temperature=temperature,
        kl_tau=kl_tau,
        top_k=top_k,
    )
    scores = compute_layer_scores(infos, kl_tau=kl_tau)

    fused = fuse_maps(maps, weights)

    if return_details:
        return fused, {
            "layers": list(layers),
            "weights": weights,
            "scores": scores,
            "selected_layers": get_selected_layers(layers, weights),
            "top_k": top_k,
            "infos": infos,
            "maps": maps,
        }

    return fused


def text_heatmap_eviba_multilayer(
    text_t,
    image_t,
    model,
    layers=(7, 8, 9),
    beta=0.1,
    var=1,
    lr=1,
    train_steps=10,
    progbar=False,
    noise_mode=None,
    sigma_fixed=1.0,
    sigma_min=0.05,
    sigma_max=1.0,
    gamma_init=0.5,
    learnable_gamma=True,
    residual_mode="centered",
    temperature=0.5,
    kl_tau=1.0,
    top_k=5,
    return_details=False,
):
    maps = []
    infos = []

    for layer_idx in layers:
        hmap, info = text_heatmap_residual_iba(
            text_t=text_t,
            image_t=image_t,
            model=model,
            layer_idx=layer_idx,
            beta=beta,
            var=var,
            lr=lr,
            train_steps=train_steps,
            progbar=progbar,
            noise_mode=noise_mode,
            sigma_fixed=sigma_fixed,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            gamma_init=gamma_init,
            learnable_gamma=learnable_gamma,
            residual_mode=residual_mode,
            return_info=True,
        )

        info["layer"] = layer_idx
        maps.append(hmap)
        infos.append(info)

    weights = compute_layer_weights(
        infos,
        temperature=temperature,
        kl_tau=kl_tau,
        top_k=top_k,
    )
    scores = compute_layer_scores(infos, kl_tau=kl_tau)

    fused = fuse_maps(maps, weights)

    if return_details:
        return fused, {
            "layers": list(layers),
            "weights": weights,
            "scores": scores,
            "selected_layers": get_selected_layers(layers, weights),
            "top_k": top_k,
            "infos": infos,
            "maps": maps,
        }

    return fused




def eviba(
    model,
    text_ids,
    image_feat,
    layers=(7, 8, 9),
    beta=0.1,
    var=1,
    lr=1,
    train_steps=10,
    noise_mode=None,
    sigma_fixed=1.0,
    sigma_min=0.05,
    sigma_max=1.0,
    gamma_init=0.5,
    learnable_gamma=True,
    residual_mode="centered",
    temperature=0.5,
    kl_tau=1.0,
    top_k=5,
    return_details=False,
):
    saliency_v = []
    saliency_t = []

    details_v = []
    details_t = []

    for idx in range(image_feat.shape[0]):
        t_id = text_ids[idx]
        i_feat = image_feat[idx:idx + 1]

        vmap, vdetail = vision_heatmap_eviba_multilayer(
            text_t=t_id,
            image_t=i_feat,
            model=model,
            layers=layers,
            beta=beta,
            var=var,
            lr=lr,
            train_steps=train_steps,
            noise_mode=noise_mode,
            sigma_fixed=sigma_fixed,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            gamma_init=gamma_init,
            learnable_gamma=learnable_gamma,
            residual_mode=residual_mode,
            temperature=temperature,
            kl_tau=kl_tau,
            top_k=top_k,
            return_details=True,
        )

        tmap, tdetail = text_heatmap_eviba_multilayer(
            text_t=t_id,
            image_t=i_feat,
            model=model,
            layers=layers,
            beta=beta,
            var=var,
            lr=lr,
            train_steps=train_steps,
            noise_mode=noise_mode,
            sigma_fixed=sigma_fixed,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            gamma_init=gamma_init,
            learnable_gamma=learnable_gamma,
            residual_mode=residual_mode,
            temperature=temperature,
            kl_tau=kl_tau,
            top_k=top_k,
            return_details=True,
        )

        saliency_v.append(vmap)
        saliency_t.append(tmap)

        details_v.append(vdetail)
        details_t.append(tdetail)

    saliency_v = np.stack(saliency_v, axis=0)

    if return_details:
        return saliency_v, saliency_t, {
            "vision": details_v,
            "text": details_t,
        }

    return saliency_v, saliency_t

