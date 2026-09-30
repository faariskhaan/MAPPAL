"""Train MAPPAL's own zone classifier: MobileNetV2 + transfer learning on OUR photos.

Self-contained on purpose (imports nothing from the repo), so it runs on Google Colab
with just this file and the dataset:

    python train_zone.py --data dataset --train walk1 --val walk2 --test test \
        --out models/zone_model.pt                      # with regularisation ("after")
    python train_zone.py --data dataset --train walk1 --val walk2 --test test \
        --out models/zone_model_noreg.pt --no-reg       # without ("before")

Dataset layout: dataset/<session>/<class>/*.jpg  (made by tools/collect_dataset.py)
Train, validation and test are SEPARATE SESSIONS, never random frames of one walk:
frames of one walk look almost identical, so a random split would let the model
"see the exam questions" and the accuracy would be fake.

Outputs:
    <out>                              checkpoint: state_dict + class names + preprocessing
    results/<run_name>/curves.png      loss and accuracy, train vs validation
    results/<run_name>/confusion_matrix.png   on the test session
    results/<run_name>/metrics.json    train/val/test accuracy, per-class precision/recall/F1
"""

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# Preferred class order (zone ids 0-3, then Other). Extra folders are added after these.
CLASS_ORDER = ["Desk", "Shelf", "Door", "Window", "Other"]
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]   # ImageNet normalisation (the backbone was trained with it)
STD = [0.229, 0.224, 0.225]
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

EXPLANATIONS = """
Regularisation used in this run, in plain words:
  * Data augmentation - every epoch the photos are randomly cropped, flipped, rotated a bit
    and made brighter/darker. The model never sees exactly the same picture twice, so it
    must learn the zone itself, not memorise pixels.
  * Dropout 0.3 - during training 30% of the features going into the last layer are
    switched off at random, so the model cannot depend on one single clue.
  * Weight decay 1e-4 (AdamW) - a small penalty on big weights keeps the model simple.
  * Label smoothing 0.1 - the target is "90% sure" instead of "100% sure", so the model
    does not become over-confident on our small dataset.
  * Early stopping (patience 5) - we watch the validation loss and keep the best epoch;
    if it has not improved for 5 epochs we stop before the model starts memorising.
"""


# --- data -------------------------------------------------------------------

def discover_classes(data_dir, sessions):
    """Class names = folder names found in the sessions, in CLASS_ORDER first."""
    found = set()
    for s in sessions:
        session_dir = Path(data_dir) / s
        if not session_dir.is_dir():
            raise SystemExit(f"Session folder not found: {session_dir}")
        found |= {p.name for p in session_dir.iterdir() if p.is_dir()}
    ordered = [c for c in CLASS_ORDER if c in found]
    return ordered + sorted(found - set(ordered))


class SessionDataset(Dataset):
    """All photos of the given sessions. Labels come from the class folder names."""

    def __init__(self, data_dir, sessions, class_names, transform):
        self.transform = transform
        self.samples = []
        for s in sessions:
            for label, name in enumerate(class_names):
                folder = Path(data_dir) / s / name
                if not folder.is_dir():
                    print(f"  warning: session '{s}' has no '{name}' folder")
                    continue
                self.samples += [(str(p), label) for p in sorted(folder.iterdir())
                                 if p.suffix.lower() in IMAGE_EXTS]
        if not self.samples:
            raise SystemExit(f"No images found in sessions {sessions} under {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.transform(Image.open(path).convert("RGB")), label


def eval_transform():
    """Exactly what the app does at run time: resize to 224x224 + normalise."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def train_transform(regularise):
    if not regularise:
        return eval_transform()
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


# --- model ------------------------------------------------------------------

def build_model(num_classes, dropout, pretrained=True):
    """MobileNetV2 backbone (ImageNet) + our own head: Dropout + Linear."""
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.last_channel, num_classes))
    return model


def set_phase(model, phase):
    """Phase 1: only the new head learns. Phase 2: also the last 2 feature blocks."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    if phase == 2:
        for p in model.features[-2:].parameters():
            p.requires_grad = True


def train_mode(model, phase):
    """Train mode, but frozen blocks stay in eval mode so their BatchNorm statistics
    (learned on millions of ImageNet photos) are not overwritten by our small dataset."""
    model.train()
    frozen = model.features if phase == 1 else model.features[:-2]
    frozen.eval()


def make_optimizer(model, lr, weight_decay):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


# --- evaluation helpers (also used by training/evaluate.py) -------------------

def confusion(y_true, y_pred, num_classes):
    """cm[t][p] = how many photos of true class t were predicted as p."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def per_class_metrics(cm, class_names):
    """Precision, recall and F1 per class from a confusion matrix."""
    out = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        predicted, actual = cm[:, i].sum(), cm[i, :].sum()
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[name] = {"precision": round(float(precision), 4), "recall": round(float(recall), 4),
                     "f1": round(float(f1), 4), "support": int(actual)}
    return out


def plot_confusion(cm, class_names, path, title="Confusion matrix (test session)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted zone")
    ax.set_ylabel("True zone")
    ax.set_title(title)
    for t in range(len(class_names)):
        for p in range(len(class_names)):
            ax.text(p, t, cm[t, p], ha="center", va="center",
                    color="white" if cm[t, p] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_curves(history, path, run_name, phase2_epoch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, label in ((a1, "loss", "Loss"), (a2, "acc", "Accuracy")):
        ax.plot(epochs, history[f"train_{key}"], "o-", label="train")
        ax.plot(epochs, history[f"val_{key}"], "o-", label="validation")
        if phase2_epoch and phase2_epoch <= len(history["train_loss"]):
            ax.axvline(phase2_epoch - 0.5, color="grey", linestyle="--", label="unfreeze")
        ax.set_xlabel("Epoch")
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.set_title(f"{label} - {run_name}")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def predict_all(model, loader, device, criterion=None):
    """Returns (y_true, y_pred, mean loss) over a loader."""
    model.eval()
    y_true, y_pred, total_loss, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if criterion is not None:
            total_loss += criterion(logits, y).item() * len(y)
        n += len(y)
        y_true += y.tolist()
        y_pred += logits.argmax(1).tolist()
    return y_true, y_pred, (total_loss / n if n else 0.0)


def accuracy(y_true, y_pred):
    return float(np.mean(np.array(y_true) == np.array(y_pred))) if y_true else 0.0


# --- training -----------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, phase):
    train_mode(model, phase)
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_comparison(results_dir):
    """If a regularised and a --no-reg run both exist, print one comparison line each."""
    runs = []
    for f in sorted(Path(results_dir).glob("*/metrics.json")):
        m = json.loads(f.read_text())
        if "test_acc" in m:
            runs.append(m)
    reg = [m for m in runs if m.get("regularised")]
    noreg = [m for m in runs if not m.get("regularised")]
    if reg and noreg:
        print("\nComparison (train accuracy vs test accuracy - the gap is overfitting):")
        for m in (reg[-1], noreg[-1]):
            label = "WITH regularisation   " if m["regularised"] else "WITHOUT regularisation"
            gap = (m["train_acc"] - m["test_acc"]) * 100
            print(f"  {label}: train {m['train_acc']:.1%}  val {m['val_acc']:.1%}  "
                  f"test {m['test_acc']:.1%}  (gap {gap:.1f} points)  [{m['run_name']}]")


def main():
    ap = argparse.ArgumentParser(description="Train the MAPPAL zone classifier")
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--train", nargs="+", default=["walk1"], help="training session(s)")
    ap.add_argument("--val", nargs="+", default=["walk2"], help="validation session(s)")
    ap.add_argument("--test", nargs="*", default=["test"], help="test session(s), never trained on")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--head-epochs", type=int, default=8, help="phase 1 length (frozen backbone)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="models/zone_model.pt")
    ap.add_argument("--results", default="results")
    ap.add_argument("--no-reg", action="store_true", help="the 'before' run: no regularisation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    seed_everything(args.seed)
    regularise = not args.no_reg
    device = "cuda" if torch.cuda.is_available() else "cpu"
    workers = args.workers if args.workers is not None else (2 if device == "cuda" else 0)
    run_name = Path(args.out).stem
    run_dir = Path(args.results) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    class_names = discover_classes(args.data, args.train + args.val + args.test)
    print(f"Run '{run_name}' on {device} | regularisation {'ON' if regularise else 'OFF'}")
    print(f"Classes: {class_names}")
    train_ds = SessionDataset(args.data, args.train, class_names, train_transform(regularise))
    train_eval_ds = SessionDataset(args.data, args.train, class_names, eval_transform())
    val_ds = SessionDataset(args.data, args.val, class_names, eval_transform())
    test_ds = SessionDataset(args.data, args.test, class_names, eval_transform()) if args.test else None
    print(f"Images: train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds) if test_ds else 0}")

    loader = lambda ds, shuffle: DataLoader(ds, batch_size=args.batch, shuffle=shuffle,  # noqa: E731
                                            num_workers=workers)
    train_loader = loader(train_ds, True)

    dropout = 0.3 if regularise else 0.0
    weight_decay = 1e-4 if regularise else 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1 if regularise else 0.0)
    model = build_model(len(class_names), dropout).to(device)

    phase = 1
    set_phase(model, phase)
    optimizer = make_optimizer(model, 1e-3, weight_decay)
    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best_loss, best_state, best_epoch, bad_epochs = float("inf"), None, 0, 0
    patience = 5
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        if epoch == args.head_epochs + 1:
            phase = 2
            set_phase(model, phase)
            optimizer = make_optimizer(model, 1e-4, weight_decay)
            bad_epochs = 0   # give fine-tuning a fair chance before early stopping
            print("-- phase 2: unfreezing the last 2 feature blocks, lr 1e-4")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, phase)
        yv, pv, va_loss = predict_all(model, loader(val_ds, False), device, criterion)
        va_acc = accuracy(yv, pv)
        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc)):
            history[k].append(round(v, 4))
        print(f"epoch {epoch:2d} | train loss {tr_loss:.3f} acc {tr_acc:.1%} | "
              f"val loss {va_loss:.3f} acc {va_acc:.1%}")

        if regularise:
            if va_loss < best_loss:
                best_loss, best_epoch, bad_epochs = va_loss, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"Early stopping: validation loss has not improved for {patience} epochs.")
                    break

    if regularise and best_state is not None:
        model.load_state_dict(best_state)
        print(f"Keeping the best epoch: {best_epoch} (lowest validation loss)")
    else:
        best_epoch = len(history["train_loss"])

    # Final, fair numbers: every split measured WITHOUT augmentation.
    yt, pt, _ = predict_all(model, loader(train_eval_ds, False), device)
    yv, pv, _ = predict_all(model, loader(val_ds, False), device)
    metrics = {
        "run_name": run_name,
        "regularised": regularise,
        "classes": class_names,
        "train_sessions": args.train, "val_sessions": args.val, "test_sessions": args.test,
        "images": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds) if test_ds else 0},
        "epochs_run": len(history["train_loss"]), "best_epoch": best_epoch,
        "train_acc": round(accuracy(yt, pt), 4),
        "val_acc": round(accuracy(yv, pv), 4),
        "history": history,
        "train_minutes": round((time.time() - start) / 60, 2),
        "device": device,
    }
    if test_ds:
        ys, ps, _ = predict_all(model, loader(test_ds, False), device)
        cm = confusion(ys, ps, len(class_names))
        metrics["test_acc"] = round(accuracy(ys, ps), 4)
        metrics["per_class_test"] = per_class_metrics(cm, class_names)
        metrics["confusion_matrix_test"] = cm.tolist()
        plot_confusion(cm, class_names, run_dir / "confusion_matrix.png")

    plot_curves(history, run_dir / "curves.png", run_name, args.head_epochs + 1)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "arch": "mobilenet_v2",
        "image_size": IMAGE_SIZE,
        "mean": MEAN,
        "std": STD,
        "dropout": dropout,
        "regularised": regularise,
        "run_name": run_name,
    }, args.out)

    print(f"\nSaved model to {args.out} and results to {run_dir}/")
    print(f"Accuracy: train {metrics['train_acc']:.1%} | val {metrics['val_acc']:.1%}"
          + (f" | test {metrics['test_acc']:.1%}" if test_ds else ""))
    print_comparison(args.results)
    if regularise:
        print(EXPLANATIONS)
    else:
        print("\nThis was the 'before' run without regularisation. Compare its train/test gap"
              " with the regularised run: a big gap = the model memorised the training walk.")


if __name__ == "__main__":
    main()
