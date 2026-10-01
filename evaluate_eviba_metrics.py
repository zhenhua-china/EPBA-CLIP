import argparse
import copy
import json
import os
import random
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets import (
    ConceptualCaptions,
    Flickr8kDataset,
    ImagenetDataset,
    collate_fn_cc,
    collate_fn_flickr8k,
    collect_fn_imagenet,
)
from eviba import eviba


warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parse_layers(value):
    value = value.strip()
    if "-" in value:
        start, end = value.split("-", 1)
        return tuple(range(int(start), int(end) + 1))
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def parse_noise_mode(value):
    if value is None:
        return None
    value = value.lower()
    if value in ("none", "null", "off"):
        return None
    return value


def normalize_visual_maps(vmaps):
    vmaps = np.asarray(vmaps)
    vmaps = np.squeeze(vmaps)

    if vmaps.ndim == 2:
        vmaps = np.expand_dims(vmaps, axis=0)

    if vmaps.ndim != 3:
        raise ValueError(f"Expected visual saliency maps with shape [B,H,W], got {vmaps.shape}")

    return np.nan_to_num(vmaps, nan=0.0, posinf=1.0, neginf=0.0)


def normalize_text_maps(tmaps):
    fixed = []
    for tmap in tmaps:
        tmap = np.asarray(tmap).squeeze()
        if tmap.ndim != 1:
            raise ValueError(f"Expected one text saliency vector per sample, got {tmap.shape}")
        fixed.append(np.nan_to_num(tmap, nan=0.0, posinf=1.0, neginf=0.0))
    return fixed


def cosine_score(output, target_feature):
    return torch.nn.CosineSimilarity(eps=1e-6)(output, target_feature)


def evaluate_image_metrics(model, image_feat, text_feature, vmap):
    cam = np.asarray(vmap).squeeze()
    if cam.ndim != 2:
        raise ValueError(f"Expected one visual saliency map with shape [H,W], got {cam.shape}")

    cam = torch.as_tensor(cam, dtype=image_feat.dtype, device=image_feat.device)

    with torch.no_grad():
        original = cosine_score(model.get_image_features(image_feat), text_feature)
        perturbed = image_feat * cam
        after = cosine_score(model.get_image_features(perturbed), text_feature)

    diff = after - original
    return {
        "vdrop": float(torch.clamp(-diff, min=0.0).item() * 100),
        "vincr": float((diff > 0).float().item() * 100),
    }


def evaluate_text_metrics(model, text_id, image_feature, tmap):
    tmap = np.asarray(tmap).squeeze()
    if tmap.ndim != 1:
        raise ValueError(f"Expected one text saliency vector, got {tmap.shape}")

    if tmap.shape[0] == text_id.shape[1]:
        tmap_inner = tmap[1:-1]
    elif tmap.shape[0] == text_id.shape[1] - 2:
        tmap_inner = tmap
    else:
        raise ValueError(
            f"Text saliency length {tmap.shape[0]} does not match token length {text_id.shape[1]}"
        )

    text_inner = text_id[:, 1:-1]
    model_clone = copy.deepcopy(model)

    with torch.no_grad():
        for idx, token_id in enumerate(text_inner[0]):
            token_id = int(token_id.item())
            scale = float(tmap_inner[idx])
            embedding = model_clone.text_model.embeddings.token_embedding.weight[token_id]
            model_clone.text_model.embeddings.token_embedding.weight[token_id] = embedding * scale

        original = cosine_score(model.get_text_features(text_inner), image_feature)
        after = cosine_score(model_clone.get_text_features(text_inner), image_feature)

    diff = after - original
    return {
        "tdrop": float(torch.clamp(-diff, min=0.0).item() * 100),
        "tincr": float((diff > 0).float().item() * 100),
    }


def metric_evaluation(model, image_feats, image_features, text_ids, text_features, saliency_v, saliency_t):
    device = next(model.parameters()).device
    all_results = []

    for image_feat, image_feature, text_id, text_feature, vmap, tmap in zip(
        image_feats,
        image_features,
        text_ids,
        text_features,
        saliency_v,
        saliency_t,
    ):
        image_feat = image_feat.unsqueeze(0).to(device)
        image_feature = image_feature.unsqueeze(0).to(device)
        text_feature = text_feature.unsqueeze(0).to(device)
        text_id = text_id.to(device)

        result = {}
        result.update(evaluate_image_metrics(model, image_feat, text_feature, vmap))
        result.update(evaluate_text_metrics(model, text_id, image_feature, tmap))
        all_results.append(result)

    return all_results


def build_dataset(args, processor):
    if args.dataset == "cc":
        dataset = ConceptualCaptions(args.cc_csv, image_preprocessor=processor)
        collate_fn = collate_fn_cc
    elif args.dataset == "imagenet":
        dataset = ImagenetDataset(
            args.imagenet_root,
            image_preprocessor=processor,
            split=args.imagenet_split,
        )
        collate_fn = collect_fn_imagenet
    elif args.dataset == "flickr8k":
        dataset = Flickr8kDataset(
            args.flickr_root,
            args.flickr_ann,
            image_preprocessor=processor,
        )
        collate_fn = collate_fn_flickr8k
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    base_dataset = dataset
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    return base_dataset, dataloader


def choose_flickr_captions(captions, batch_xs, tokenizer, model, device):
    if not any(isinstance(cp, (list, tuple)) for cp in captions):
        return list(captions)

    loss_fn = torch.nn.CosineSimilarity(eps=1e-6)
    batch_xs = batch_xs.to(device)

    with torch.no_grad():
        image_features = model.get_image_features(batch_xs)

    selected = []
    for idx, cp in enumerate(captions):
        if isinstance(cp, str):
            selected.append(cp)
            continue

        text_ids = [
            torch.tensor([tokenizer.encode(c, add_special_tokens=True)], device=device)
            for c in cp
        ]
        with torch.no_grad():
            text_features = torch.cat([model.get_text_features(t) for t in text_ids], dim=0)
            scores = loss_fn(image_features[idx : idx + 1], text_features)

        selected.append(cp[int(scores.argmax().item())])

    return selected


def encode_captions(captions, tokenizer, model, device):
    text_ids = [
        torch.tensor([tokenizer.encode(cp, add_special_tokens=True)], device=device)
        for cp in captions
    ]
    with torch.no_grad():
        text_features = torch.cat([model.get_text_features(t) for t in text_ids], dim=0)
    return text_ids, text_features


def summarize(results):
    return {
        "vdrop": float(sum(r["vdrop"] for r in results) / len(results)),
        "vincr": float(sum(r["vincr"] for r in results) / len(results)),
        "tdrop": float(sum(r["tdrop"] for r in results) / len(results)),
        "tincr": float(sum(r["tincr"] for r in results) / len(results)),
    }


def get_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate the EviBA method with vdrop/vincr/tdrop/tincr."
    )

    parser.add_argument("--dataset", type=str, default="cc", choices=["cc", "imagenet", "flickr8k"])
    parser.add_argument("--clip_path", type=str, default=os.path.join("models", "clip-vit-base-patch32"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", type=str, default=None)

    parser.add_argument("--cc_csv", type=str, default=os.path.join("datasets", "cc.csv"))
    parser.add_argument("--imagenet_root", type=str, default=os.path.join("datasets", "tiny-imagenet-200"))
    parser.add_argument("--imagenet_split", type=str, default="val", choices=["val", "train"])
    parser.add_argument("--flickr_root", type=str, default="datasets")
    parser.add_argument("--flickr_ann", type=str, default=os.path.join("datasets", "en_val.json"))

    parser.add_argument("--layers", type=str, default="0-11")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--var", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--train_steps", type=int, default=10)

    parser.add_argument(
        "--noise_mode",
        type=str,
        default="patchwise",
        choices=["none", "fixed", "learnable", "patchwise"],
    )
    parser.add_argument("--sigma_fixed", type=float, default=0.1)
    parser.add_argument("--sigma_min", type=float, default=0.05)
    parser.add_argument("--sigma_max", type=float, default=1.0)
    parser.add_argument("--gamma_init", type=float, default=0.5)
    parser.add_argument("--freeze_gamma", action="store_true")
    parser.add_argument(
        "--residual_mode",
        type=str,
        default="centered",
        choices=["centered", "detached", "token_centered", "channel_centered", "zero"],
    )
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--kl_tau", type=float, default=1.0)

    return parser


def main():
    args = get_parser().parse_args()
    setup_seed(args.seed)

    from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers = parse_layers(args.layers)
    noise_mode = parse_noise_mode(args.noise_mode)

    model = CLIPModel.from_pretrained(args.clip_path, local_files_only=True).to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_path, local_files_only=True)
    tokenizer = CLIPTokenizerFast.from_pretrained(args.clip_path, local_files_only=True)
    model.eval()

    base_dataset, dataloader = build_dataset(args, processor)

    print("Evaluating EviBA")
    print(f"dataset={args.dataset}, samples={len(dataloader.dataset)}, batch_size={args.batch_size}")
    print(f"layers={layers}, beta={args.beta}, train_steps={args.train_steps}, noise_mode={noise_mode}")

    image_feats = []
    image_features = []
    text_ids = []
    text_features = []
    image_saliency = []
    text_saliency = []

    pbar = tqdm(total=len(dataloader))
    for _, captions, batch_xs in dataloader:
        if isinstance(base_dataset, Flickr8kDataset):
            captions = choose_flickr_captions(captions, batch_xs, tokenizer, model, device)
        else:
            captions = list(captions)

        batch_xs = batch_xs.to(device)
        batch_text_ids, batch_text_features = encode_captions(captions, tokenizer, model, device)

        with torch.no_grad():
            batch_image_features = model.get_image_features(batch_xs).detach().cpu()

        v_saliency, t_saliency = eviba(
            model=model,
            text_ids=batch_text_ids,
            image_feat=batch_xs,
            layers=layers,
            beta=args.beta,
            var=args.var,
            lr=args.lr,
            train_steps=args.train_steps,
            noise_mode=noise_mode,
            sigma_fixed=args.sigma_fixed,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            gamma_init=args.gamma_init,
            learnable_gamma=not args.freeze_gamma,
            residual_mode=args.residual_mode,
            temperature=args.temperature,
            kl_tau=args.kl_tau,
            return_details=False,
        )

        image_saliency.append(normalize_visual_maps(v_saliency))
        text_saliency.extend(normalize_text_maps(t_saliency))
        image_feats.append(batch_xs.detach().cpu())
        image_features.append(batch_image_features)
        text_ids.extend(batch_text_ids)
        text_features.extend(batch_text_features.detach().cpu())

        pbar.update(1)

    pbar.close()

    image_feats = torch.cat(image_feats, dim=0)
    image_features = torch.cat(image_features, dim=0)
    text_features = torch.stack(text_features, dim=0)
    image_saliency = np.concatenate(image_saliency, axis=0)

    results = metric_evaluation(
        model,
        image_feats,
        image_features,
        text_ids,
        text_features,
        image_saliency,
        text_saliency,
    )
    metrics = summarize(results)

    print("vdrop(Img Conf Drop):", metrics["vdrop"])
    print("vincr(Img Conf Incr):", metrics["vincr"])
    print("tdrop(Text Conf Drop):", metrics["tdrop"])
    print("tincr(Text Conf Incr):", metrics["tincr"])

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metrics": metrics,
                    "num_samples": len(results),
                    "args": vars(args),
                    "layers": list(layers),
                    "noise_mode": noise_mode,
                },
                f,
                indent=2,
            )
        print(f"Saved metrics to {args.output_json}")


if __name__ == "__main__":
    main()

