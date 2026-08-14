"""Train the causal temporal Transformer on cached feature windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from egovision.data.cached import CachedWindowDataset, split_by_subject
from egovision.models.causal_transformer import CausalTemporalTransformer
from egovision.metrics import balanced_frame_accuracy, confusion_matrix, majority_class_accuracy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/features"))
    parser.add_argument("--held-out-subject", default="S4")
    parser.add_argument("--num-classes", type=int, default=73)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/causal_transformer"))
    return parser


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    paths = tuple(sorted(args.features.glob("*.pt")))
    train_paths, test_paths = split_by_subject(paths, args.held_out_subject)
    train_data = CachedWindowDataset(train_paths, context=args.context)
    test_data = CachedWindowDataset(test_paths, context=args.context)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=lambda batch: batch[0])
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=lambda batch: batch[0])

    model = CausalTemporalTransformer(384, args.num_classes, max_context=args.context)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for features, labels in train_loader:
            logits = model(features.unsqueeze(0))
            loss = loss_fn(logits.reshape(-1, args.num_classes), labels.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch={epoch + 1} train_loss={total_loss / len(train_data):.4f}")

    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for features, labels in test_loader:
            logits = model(features.unsqueeze(0)).squeeze(0)
            valid = labels != -1
            all_logits.append(logits[valid])
            all_labels.append(labels[valid])
    test_logits = torch.cat(all_logits)
    test_labels = torch.cat(all_labels)
    predictions = test_logits.argmax(dim=-1)
    accuracy = float((predictions == test_labels).float().mean())
    balanced = balanced_frame_accuracy(test_logits, test_labels)
    majority = majority_class_accuracy(test_labels)
    matrix = confusion_matrix(test_logits, test_labels, args.num_classes)
    result = {
        "held_out_subject": args.held_out_subject,
        "seed": args.seed,
        "epochs": args.epochs,
        "context": args.context,
        "num_classes": args.num_classes,
        "train_windows": len(train_data),
        "test_windows": len(test_data),
        "test_frames": len(test_labels),
        "test_accuracy": accuracy,
        "balanced_accuracy": balanced,
        "majority_accuracy": majority,
        "class_support": torch.bincount(test_labels, minlength=args.num_classes).tolist(),
        "confusion_matrix": matrix.tolist(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / f"held_out_{args.held_out_subject}.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(
        f"held_out={args.held_out_subject} accuracy={accuracy:.4f} "
        f"balanced_accuracy={balanced:.4f} majority_accuracy={majority:.4f}"
    )
    print(f"results={output_path}")


if __name__ == "__main__":
    run(build_parser().parse_args())
