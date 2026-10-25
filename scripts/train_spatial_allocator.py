"""Train the learned spatial-importance allocator on synthetic patch data.

This runner verifies the spatial-allocation recipe the design specifies.
It runs on CPU with synthetic DINOv2-style patch-token clips (no data
download). Each clip is a sequence of frames in which a 4x4-cell
hand-object box carries the action signal and moves over time, while
the background carries distractor context and a weak mixture of other
class signals (other objects in the scene). Action labels are random
per segment, so the only evidence for the action is inside the box.

Training has three stages:

1. KL pre-training: the importance map is supervised with KL
   divergence against the normalized hand-box target map, so the
   allocator localizes first.
2. Joint training: action cross-entropy (through the causal
   Transformer) plus the weighted KL term, the joint objective the
   design specifies.
3. Shared temporal model: one causal Transformer is trained on the
   frozen allocator's features and then evaluated under two input
   policies, ``learned`` (localized features) and ``uniform`` (plain
   patch mean), so the comparison isolates the input policy.

Held-out clips never enter training. The report compares both
policies and the fraction of predicted importance mass inside the
ground-truth box.

Usage (CPU):
    python3 -m scripts.train_spatial_allocator
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from egovision.data.synthetic import make_signal_patch_sequence
from egovision.models import (
    CausalTemporalTransformer,
    SpatialImportanceAllocator,
    box_to_importance_map,
    importance_kl_loss,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=int, default=8)
    parser.add_argument("--held-out-clips", type=int, default=3)
    parser.add_argument("--length", type=int, default=48)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--segment-length", type=int, default=12)
    parser.add_argument("--feature-dim", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--kl-pretrain-epochs", type=int, default=30)
    parser.add_argument("--transformer-epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--kl-weight", type=float, default=0.5)
    parser.add_argument("--signal-std", type=float, default=4.0)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--distractor-std", type=float, default=0.5)
    parser.add_argument("--background-mixture-std", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/spatial_allocator"))
    return parser


def _clips(args: argparse.Namespace) -> list:
    return [
        make_signal_patch_sequence(
            length=args.length,
            grid=16,
            patch_dim=args.feature_dim,
            num_classes=args.num_classes,
            segment_length=args.segment_length,
            signal_std=args.signal_std,
            noise_std=args.noise_std,
            distractor_std=args.distractor_std,
            background_mixture_std=args.background_mixture_std,
            seed=args.seed + index,
        )
        for index in range(args.clips)
    ]


def _train(
    clips: list,
    args: argparse.Namespace,
) -> SpatialImportanceAllocator:
    """Jointly train the allocator and a causal Transformer.

    The allocator receives action cross-entropy (through the
    Transformer) plus the weighted KL term against the box target
    maps; the Transformer receives the action cross-entropy.
    """
    allocator = _pretrain_allocator_kl(clips, args, args.kl_pretrain_epochs)
    transformer = CausalTemporalTransformer(
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
        max_context=args.length,
    )
    optimizer = torch.optim.AdamW(
        list(allocator.parameters()) + list(transformer.parameters()),
        lr=args.learning_rate,
    )
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        total = 0.0
        for sample in clips:
            patches = sample.patches.unsqueeze(0)  # [1, T, 256, D]
            weights, features = allocator(patches)
            logits = transformer(features)
            targets = torch.stack([
                box_to_importance_map(box, grid=16, width=256, height=256)
                for box in sample.boxes
            ]).reshape(1, -1, 256)
            action_loss = loss_fn(
                logits.reshape(-1, args.num_classes), sample.labels.unsqueeze(0).reshape(-1)
            )
            kl_loss = importance_kl_loss(weights.reshape(-1, 256), targets.reshape(-1, 256))
            loss = action_loss + args.kl_weight * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch={epoch + 1} loss={total / len(clips):.4f}")

    allocator.eval()
    return allocator


def _pretrain_allocator_kl(
    clips: list,
    args: argparse.Namespace,
    epochs: int,
) -> SpatialImportanceAllocator:
    """Fit the importance maps to the box targets before joint training.

    This is the localization stage of the design: the map is supervised
    with KL divergence against the normalized hand-box target, so the
    allocator already localizes when the recognition objective is
    added.
    """
    allocator = SpatialImportanceAllocator(feature_dim=args.feature_dim, grid=16)
    optimizer = torch.optim.AdamW(allocator.parameters(), lr=args.learning_rate)
    for epoch in range(epochs):
        total = 0.0
        for sample in clips:
            patches = sample.patches.unsqueeze(0)  # [1, T, 256, D]
            weights, _ = allocator(patches)
            targets = torch.stack([
                box_to_importance_map(box, grid=16, width=256, height=256)
                for box in sample.boxes
            ]).reshape(1, -1, 256)
            loss = importance_kl_loss(weights.reshape(-1, 256), targets.reshape(-1, 256))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"kl_pretrain={epoch + 1} kl_loss={total / len(clips):.4f}")
    allocator.eval()
    return allocator


def _evaluate(
    sample,
    features_fn,
    transformer: CausalTemporalTransformer,
) -> dict:
    """Evaluate one temporal model on one clip; return accuracy metrics."""
    with torch.no_grad():
        features = features_fn(sample.patches.unsqueeze(0))
        logits = transformer(features).squeeze(0)
        valid = sample.labels != -1
        logits = logits[valid]
        labels = sample.labels[valid]
        predictions = logits.argmax(dim=-1)
        accuracy = float((predictions == labels).float().mean())
    # Balanced accuracy: mean per-class recall over classes present.
    recalls = []
    for class_id in torch.unique(labels):
        mask = labels == class_id
        recalls.append(float((predictions[mask] == class_id).float().mean()))
    balanced = float(sum(recalls) / len(recalls))
    return {"accuracy": accuracy, "balanced_accuracy": balanced, "frames": int(labels.numel())}


def _evaluate_clips(
    clips: list,
    features_fn,
    transformer: CausalTemporalTransformer,
) -> dict:
    """Average per-clip metrics across several held-out clips."""
    per_clip = [_evaluate(clip, features_fn, transformer) for clip in clips]
    accuracy = float(sum(c["accuracy"] for c in per_clip) / len(per_clip))
    balanced = float(sum(c["balanced_accuracy"] for c in per_clip) / len(per_clip))
    frames = int(sum(c["frames"] for c in per_clip))
    return {"accuracy": accuracy, "balanced_accuracy": balanced, "frames": frames}


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    clips = _clips(args)
    if args.held_out_clips < 1 or args.held_out_clips >= args.clips:
        raise ValueError("held_out_clips must be in [1, clips - 1].")
    # Hold out the first N clips for testing; train on the rest. No test
    # evidence enters any training phase.
    held_out, train_clips = clips[: args.held_out_clips], clips[args.held_out_clips :]

    # 1) Joint training: allocator (KL + cross-entropy) and its
    #    Transformer trained together on the train clips.
    # _train first runs KL pre-training (map -> box targets), then the
    # joint CE + KL phase.
    allocator = _train(train_clips, args)
    # 2) Both input policies share the same temporal model: retrain one
    #    Transformer on the frozen allocator's features, then evaluate the
    #    same Transformer on learned vs uniform (patch-mean) features.
    transformer = _train_transformer_on_features(allocator, train_clips, args)

    def learned_features(patches: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, features = allocator(patches)
        return features

    def uniform_features(patches: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return patches.mean(dim=2)

    learned = _evaluate_clips(held_out, learned_features, transformer)
    uniform = _evaluate_clips(held_out, uniform_features, transformer)

    # 3) Report where the trained allocator puts its attention mass.
    with torch.no_grad():
        inside: list[float] = []
        for sample in held_out:
            weights, _ = allocator(sample.patches.unsqueeze(0))
            for t, (x0, y0, x1, y1) in enumerate(sample.boxes):
                c0, c1 = max(0, x0 // 16 - 1), min(16, (x1 + 15) // 16 + 1)
                r0, r1 = max(0, y0 // 16 - 1), min(16, (y1 + 15) // 16 + 1)
                inside.append(float(weights[0, t].reshape(16, 16)[r0:r1, c0:c1].sum()))
    mean_inside_mass = float(sum(inside) / len(inside))
    # Random baseline: a uniform map puts 1/16 of its mass in a 4x4 box.
    uniform_mass = 1.0 / 16.0

    result = {
        "synthetic": True,
        "clips": args.clips,
        "held_out_clips": args.held_out_clips,
        "length": args.length,
        "num_classes": args.num_classes,
        "epochs": args.epochs,
        "kl_weight": args.kl_weight,
        "kl_pretrain_epochs": args.kl_pretrain_epochs,
        "transformer_epochs": args.transformer_epochs,
        "signal_std": args.signal_std,
        "noise_std": args.noise_std,
        "distractor_std": args.distractor_std,
        "background_mixture_std": args.background_mixture_std,
        "seed": args.seed,
        "train_clips": len(train_clips),
        "allocation_inside_box_mass": mean_inside_mass,
        "uniform_mass_baseline": uniform_mass,
        "mass_ratio_vs_uniform": mean_inside_mass / uniform_mass,
        "learned": learned,
        "uniform": uniform,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "synthetic_comparison.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(
        f"learned accuracy={learned['accuracy']:.4f} "
        f"balanced={learned['balanced_accuracy']:.4f}"
    )
    print(
        f"uniform accuracy={uniform['accuracy']:.4f} "
        f"balanced={uniform['balanced_accuracy']:.4f}"
    )
    print(
        f"allocation inside-box mass={mean_inside_mass:.4f} "
        f"(uniform baseline {uniform_mass:.4f}, "
        f"ratio {mean_inside_mass / uniform_mass:.2f}x)"
    )
    print(f"results={output_path}")


def _train_transformer_on_features(
    allocator: SpatialImportanceAllocator | None,
    clips: list,
    args: argparse.Namespace,
) -> CausalTemporalTransformer:
    """Train a fresh causal Transformer on features from a fixed policy."""
    transformer = CausalTemporalTransformer(
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
        max_context=args.length,
    )
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(args.transformer_epochs):
        for sample in clips:
            patches = sample.patches.unsqueeze(0)
            if allocator is None:
                features = patches.mean(dim=2)
            else:
                with torch.no_grad():
                    _, features = allocator(patches)
            logits = transformer(features)
            loss = loss_fn(
                logits.reshape(-1, args.num_classes), sample.labels.unsqueeze(0).reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    transformer.eval()
    return transformer


if __name__ == "__main__":
    run(build_parser().parse_args())
