"""Train and evaluate an EGTEA full-frame baseline or causal Transformer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from egovision.data.cached import CachedFrameDataset, CachedWindowDataset
from egovision.data.egtea import cache_paths_for_split, parse_split_file
from egovision.device import get_device
from egovision.metrics import balanced_frame_accuracy, confusion_matrix, majority_class_accuracy
from egovision.models.causal_transformer import CausalTemporalTransformer
from egovision.models.frame_classifier import FrameLinearClassifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--model", choices=("frame", "transformer"), default="transformer")
    parser.add_argument("--num-classes", type=int, default=106)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _evaluate(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict[str, object]:
    predictions = logits.argmax(dim=-1)
    return {
        "test_accuracy": float((predictions == labels).float().mean()),
        "balanced_accuracy": balanced_frame_accuracy(logits, labels),
        "majority_accuracy": majority_class_accuracy(labels),
        "class_support": torch.bincount(labels, minlength=num_classes).tolist(),
        "confusion_matrix": confusion_matrix(logits, labels, num_classes).tolist(),
    }


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    train_records = parse_split_file(args.train_split)
    test_records = parse_split_file(args.test_split)
    train_paths = cache_paths_for_split(train_records, args.features)
    test_paths = cache_paths_for_split(test_records, args.features)
    if args.model == "frame":
        train_data = CachedFrameDataset(train_paths)
        test_data = CachedFrameDataset(test_paths)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size)
        model: nn.Module = FrameLinearClassifier(384, args.num_classes).to(device)
    else:
        train_data = CachedWindowDataset(train_paths, context=args.context)
        test_data = CachedWindowDataset(test_paths, context=args.context)
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=lambda batch: batch[0])
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=lambda batch: batch[0])
        model = CausalTemporalTransformer(384, args.num_classes, max_context=args.context).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features.unsqueeze(1)).squeeze(1) if args.model == "frame" else model(features.unsqueeze(0))
            loss = loss_fn(logits.reshape(-1, args.num_classes), labels.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch={epoch + 1} train_loss={total_loss / len(train_loader):.4f}")

    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features.unsqueeze(1)).squeeze(1) if args.model == "frame" else model(features.unsqueeze(0)).squeeze(0)
            valid = labels != -1
            all_logits.append(logits[valid].cpu())
            all_labels.append(labels[valid].cpu())
    test_logits, test_labels = torch.cat(all_logits), torch.cat(all_labels)
    result = {
        "dataset": "EGTEA",
        "model": args.model,
        "train_split": str(args.train_split),
        "test_split": str(args.test_split),
        "seed": args.seed,
        "epochs": args.epochs,
        "context": args.context,
        "num_classes": args.num_classes,
        "train_clips": len(train_paths),
        "test_clips": len(test_paths),
        "test_frames": len(test_labels),
        **_evaluate(test_logits, test_labels, args.num_classes),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        f"model={args.model} accuracy={result['test_accuracy']:.4f} "
        f"balanced_accuracy={result['balanced_accuracy']:.4f}"
    )
    print(f"results={args.output}")


if __name__ == "__main__":
    run(build_parser().parse_args())
