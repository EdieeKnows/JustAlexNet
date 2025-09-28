from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from dataset_imagenett import train_loader, val_loader
from model_factory import build_model

logger = logging.getLogger("trainer")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _compute_topk_correct(
    predictions: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...]
) -> List[int]:
    """Return the number of correct predictions for each ``k`` in ``topk``."""

    if predictions.ndim != 2:
        raise ValueError("Predictions tensor is expected to be 2-dimensional")

    max_k = min(max(topk), predictions.size(1))
    _, pred_topk = predictions.topk(max_k, 1, True, True)
    pred_topk = pred_topk.t()

    expanded_targets = targets.view(1, -1).expand_as(pred_topk)
    correct = pred_topk.eq(expanded_targets)

    results: List[int] = []
    for k in topk:
        k = min(k, predictions.size(1))
        if k <= 0:
            results.append(0)
            continue
        correct_k = correct[:k].any(dim=0).sum().item()
        results.append(int(correct_k))
    return results


# ---------------------------------------------------------------------------
# Checkpoint and history utilities
# ---------------------------------------------------------------------------


def _initialise_history() -> Dict[str, List[float]]:
    """Return a ready-to-use container for tracking training statistics."""
    return {
        "train_loss": [],
        "train_top1": [],
        "train_top5": [],
        "val_loss": [],
        "val_top1": [],
        "val_top5": [],
    }


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip ``module.`` prefixes introduced by ``DataParallel`` if present."""
    if not state_dict:
        return state_dict
    if all(not key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def _load_model_state(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    cleaned_state = _clean_state_dict(state_dict)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(cleaned_state)
    else:
        model.load_state_dict(cleaned_state)


def _extract_model_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    """Handle checkpoints saved under different key conventions."""
    if isinstance(checkpoint, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def _prepare_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    history: Dict[str, List[float]],
    *,
    epoch: int,
    best_val_top1: float,
    best_val_loss: float,
    epochs_without_improvement: int,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> Dict[str, object]:
    """Collect training state into a serialisable dictionary."""
    model_state = (
        model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    )
    checkpoint: Dict[str, object] = {
        "epoch": epoch,
        "model_state": model_state,
        "model_state_dict": model_state,
        "optimizer_state": optimizer.state_dict(),
        "history": copy.deepcopy(history),
        "best_val_acc": best_val_top1,
        "best_val_loss": best_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
    }
    if scaler is not None:
        checkpoint["scaler_state"] = scaler.state_dict()
    return checkpoint


def _save_confusion_matrix(
    matrix: torch.Tensor, output_dir: Path, *, epoch: int
) -> Tuple[Path, Path]:
    """Persist the confusion matrix as both a tensor and a CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / f"confusion_matrix_epoch_{epoch:03d}.pt"
    csv_path = output_dir / f"confusion_matrix_epoch_{epoch:03d}.csv"
    torch.save(matrix, tensor_path)
    np.savetxt(csv_path, matrix.cpu().numpy(), fmt="%d", delimiter=",")
    return tensor_path, csv_path


def train_loop(
    dataloader,
    model,
    loss_fn,
    optimizer,
    device,
    *,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_autocast: bool = False,
):
    """Run a training epoch and return aggregated loss and accuracy."""

    size = len(dataloader.dataset)
    model.train()

    running_loss = 0.0
    running_top1 = 0
    running_top5 = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_autocast):
            pred = model(X)
            loss = loss_fn(pred, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_loss = loss.item()
        top1_correct, top5_correct = _compute_topk_correct(pred, y, (1, 5))
        batch_top1_acc = top1_correct / len(X)
        batch_top5_acc = top5_correct / len(X)

        running_loss += batch_loss * X.size(0)
        running_top1 += top1_correct
        running_top5 += top5_correct

        if batch % 100 == 0:
            current_samples = batch * dataloader.batch_size + len(X)
            avg_loss = running_loss / current_samples
            avg_top1 = running_top1 / current_samples
            avg_top5 = running_top5 / current_samples
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                (
                    "Batch {batch:>4d}: loss={loss:.4f}, top1={top1:.2f}%, top5={top5:.2f}%, "
                    "avg_loss={avg_loss:.4f}, avg_top1={avg_top1:.2f}%, avg_top5={avg_top5:.2f}%, "
                    "lr={lr:.6f} [{current}/{total}]"
                ).format(
                    batch=batch,
                    loss=batch_loss,
                    top1=batch_top1_acc * 100,
                    top5=batch_top5_acc * 100,
                    avg_loss=avg_loss,
                    avg_top1=avg_top1 * 100,
                    avg_top5=avg_top5 * 100,
                    lr=lr,
                    current=current_samples,
                    total=size,
                )
            )

    epoch_loss = running_loss / size
    epoch_top1 = running_top1 / size
    epoch_top5 = running_top5 / size

    return epoch_loss, epoch_top1, epoch_top5


def evaluate(
    dataloader,
    model,
    loss_fn,
    device,
    *,
    num_classes: int,
    use_autocast: bool = False,
):
    """Evaluate the model and return loss, accuracy metrics and confusion matrix."""

    size = len(dataloader.dataset)
    model.eval()

    running_loss = 0.0
    running_top1 = 0
    running_top5 = 0
    # Track true (rows) vs predicted (columns) class counts
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=use_autocast):
                pred = model(X)
                loss = loss_fn(pred, y)

            running_loss += loss.item() * X.size(0)
            top1_correct, top5_correct = _compute_topk_correct(pred, y, (1, 5))
            running_top1 += top1_correct
            running_top5 += top5_correct

            predicted_labels = pred.argmax(1)
            for true_label, predicted_label in zip(
                y.view(-1).cpu(), predicted_labels.view(-1).cpu()
            ):
                confusion_matrix[true_label.long(), predicted_label.long()] += 1

    epoch_loss = running_loss / size
    epoch_top1 = running_top1 / size
    epoch_top5 = running_top5 / size

    return epoch_loss, epoch_top1, epoch_top5, confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train image classification models")
    parser.add_argument(
        "--model",
        default="alexnet",
        help="Model architecture to train (alexnet, resnet18 or densenet201)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Number of output classes",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help="Number of epochs with no validation loss improvement before stopping. Use 0 to disable.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum change in validation loss to qualify as an improvement for early stopping.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory where checkpoints and confusion matrices are stored.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional path to a checkpoint file to resume training from.",
    )
    return parser.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    logging.basicConfig(filename=log_path, format="%(asctime)s %(message)s", filemode="w")
    configured_logger = logging.getLogger("trainer")
    configured_logger.setLevel(logging.DEBUG)
    return configured_logger


def ensure_checkpoint_dirs(base_dir: Path) -> Tuple[Path, Path, Path, Path]:
    checkpoint_dir = base_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir = checkpoint_dir / "confusion_matrices"
    confusion_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "best_model.pth"
    latest_model_path = checkpoint_dir / "latest_model.pth"
    return checkpoint_dir, confusion_dir, best_model_path, latest_model_path


def configure_model_and_device(
    model_name: str, num_classes: int, logger: logging.Logger
) -> Tuple[nn.Module, torch.device, bool, torch.cuda.amp.GradScaler | None]:
    # Select the most capable device available (CUDA > MPS > CPU)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using {device} device")
    logger.info(f"Selected model: {model_name}")

    model = build_model(model_name, num_classes).to(device)

    use_autocast = False
    scaler = None
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_capabilities: List[bool] = []

    # Only enable multi-GPU and AMP logic when CUDA is available
    if device.type == "cuda":
        logger.info(f"Detected {gpu_count} CUDA device(s)")
        for idx in range(gpu_count):
            capability = torch.cuda.get_device_capability(idx)
            name = torch.cuda.get_device_name(idx)
            supports_tensor_core = capability[0] >= 7
            device_capabilities.append(supports_tensor_core)
            logger.info(
                "GPU {idx}: {name} (compute capability {major}.{minor}) – Tensor Cores {support}".format(
                    idx=idx,
                    name=name,
                    major=capability[0],
                    minor=capability[1],
                    support="enabled" if supports_tensor_core else "not available",
                )
            )

        if gpu_count > 1:
            logger.info("Wrapping model with torch.nn.DataParallel for multi-GPU training")
            model = nn.DataParallel(model)

        if device_capabilities and all(device_capabilities):
            use_autocast = True
            scaler = torch.cuda.amp.GradScaler()
            logger.info(
                "All CUDA devices support Tensor Cores – enabling automatic mixed precision"
            )
        else:
            logger.info(
                "Tensor Cores not detected on all CUDA devices – training will use full precision"
            )
    else:
        logger.info("CUDA not available – running on a single device")

    return model, device, use_autocast, scaler


def resume_training_if_requested(
    resume_path: Path | None,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    history: Dict[str, List[float]],
    logger: logging.Logger,
    *,
    best_val_top1: float,
    best_val_loss: float,
    start_epoch: int,
    epochs_without_improvement: int,
) -> Tuple[int, float, float, int, Dict[str, List[float]]]:
    if resume_path is None:
        return start_epoch, best_val_top1, best_val_loss, epochs_without_improvement, history

    resolved_path = resume_path.expanduser().resolve()
    if not resolved_path.is_file():
        logger.warning(
            f"Resume checkpoint not found at {resolved_path}. Starting fresh training run."
        )
        return start_epoch, best_val_top1, best_val_loss, epochs_without_improvement, history

    logger.info(f"Resuming training from checkpoint: {resolved_path}")
    checkpoint = torch.load(resolved_path, map_location=device)
    state_dict = _extract_model_state_dict(checkpoint)
    _load_model_state(model, state_dict)

    if isinstance(checkpoint, dict):
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scaler is not None and checkpoint.get("scaler_state") is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = checkpoint.get("epoch", start_epoch)
        best_val_top1 = checkpoint.get("best_val_acc", best_val_top1)
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        epochs_without_improvement = checkpoint.get(
            "epochs_without_improvement", epochs_without_improvement
        )
        loaded_history = checkpoint.get("history")
        if loaded_history:
            # Support legacy checkpoints that stored accuracy without the top-k naming
            if "train_acc" in loaded_history and "train_top1" not in loaded_history:
                loaded_history["train_top1"] = loaded_history["train_acc"]
            if "val_acc" in loaded_history and "val_top1" not in loaded_history:
                loaded_history["val_top1"] = loaded_history["val_acc"]

            updated_history = _initialise_history()
            for key in updated_history:
                if key in loaded_history:
                    updated_history[key] = list(loaded_history[key])

            if not updated_history["train_top5"]:
                updated_history["train_top5"] = [
                    float("nan") for _ in range(len(updated_history["train_top1"]))
                ]
            if not updated_history["val_top5"]:
                updated_history["val_top5"] = [
                    float("nan") for _ in range(len(updated_history["val_top1"]))
                ]

            history = updated_history

    logger.info(
        "Checkpoint loaded. Resuming from epoch %d with best val top-1 accuracy %.2f%%",
        start_epoch + 1,
        best_val_top1 * 100,
    )
    return start_epoch, best_val_top1, best_val_loss, epochs_without_improvement, history


def plot_history(history: Dict[str, List[float]], logger: logging.Logger) -> None:
    epochs_completed = len(history["train_loss"])
    if epochs_completed == 0:
        logger.warning("No training epochs completed; skipping plot generation.")
        return

    epochs_range = range(1, epochs_completed + 1)
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, [acc * 100 for acc in history["train_top1"]], label="Train Top-1")
    plt.plot(epochs_range, [acc * 100 for acc in history["val_top1"]], label="Val Top-1")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Top-1 Accuracy over Epochs")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, [acc * 100 for acc in history["train_top5"]], label="Train Top-5")
    plt.plot(epochs_range, [acc * 100 for acc in history["val_top5"]], label="Val Top-5")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Top-5 Accuracy over Epochs")
    plt.legend()

    plot_path = Path("training_metrics.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    logger.info(f"Training metrics plot saved to {plot_path.resolve()}")


def run_training(
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    history: Dict[str, List[float]],
    *,
    use_autocast: bool,
    scaler: torch.cuda.amp.GradScaler | None,
    best_val_top1: float,
    best_val_loss: float,
    start_epoch: int,
    epochs_without_improvement: int,
    best_model_path: Path,
    latest_model_path: Path,
    confusion_dir: Path,
    logger: logging.Logger,
) -> None:
    try:
        for epoch_index in range(start_epoch, args.epochs):
            epoch_number = epoch_index + 1
            logger.info(f"\nEpoch {epoch_number}/{args.epochs}\n-------------------------------")
            train_loss, train_top1, train_top5 = train_loop(
                train_loader,
                model,
                loss_fn,
                optimizer,
                device,
                scaler=scaler,
                use_autocast=use_autocast,
            )
            val_loss, val_top1, val_top5, val_confusion = evaluate(
                val_loader,
                model,
                loss_fn,
                device,
                num_classes=args.num_classes,
                use_autocast=use_autocast,
            )

            history["train_loss"].append(train_loss)
            history["train_top1"].append(train_top1)
            history["train_top5"].append(train_top5)
            history["val_loss"].append(val_loss)
            history["val_top1"].append(val_top1)
            history["val_top5"].append(val_top5)

            confusion_tensor_path, confusion_csv_path = _save_confusion_matrix(
                val_confusion, confusion_dir, epoch=epoch_number
            )
            logger.info(
                f"Validation confusion matrix saved to {confusion_tensor_path} and {confusion_csv_path}"
            )

            should_stop = False
            if val_loss < best_val_loss - args.early_stop_min_delta:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if (
                    args.early_stop_patience > 0
                    and epochs_without_improvement >= args.early_stop_patience
                ):
                    logger.info(
                        "Early stopping triggered after %d epochs without improvement.",
                        epochs_without_improvement,
                    )
                    should_stop = True

            is_new_best = val_top1 > best_val_top1
            if is_new_best:
                best_val_top1 = val_top1
                best_val_loss = min(best_val_loss, val_loss)

            checkpoint = _prepare_checkpoint(
                model,
                optimizer,
                history,
                epoch=epoch_number,
                best_val_top1=best_val_top1,
                best_val_loss=best_val_loss,
                epochs_without_improvement=epochs_without_improvement,
                scaler=scaler,
            )
            torch.save(checkpoint, latest_model_path)

            if is_new_best:
                torch.save(checkpoint, best_model_path)
                logger.info(f"New best model saved with val_top1={val_top1 * 100:.2f}%")
            else:
                logger.info(f"Best val_top1 so far: {best_val_top1 * 100:.2f}%")

            logger.info(
                (
                    "Epoch Summary: train_loss={train_loss:.4f}, train_top1={train_top1:.2f}%, "
                    "train_top5={train_top5:.2f}%, val_loss={val_loss:.4f}, val_top1={val_top1:.2f}%, "
                    "val_top5={val_top5:.2f}%"
                ).format(
                    train_loss=train_loss,
                    train_top1=train_top1 * 100,
                    train_top5=train_top5 * 100,
                    val_loss=val_loss,
                    val_top1=val_top1 * 100,
                    val_top5=val_top5 * 100,
                )
            )

            if should_stop:
                break

        else:
            logger.info("Training complete!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Finalizing before exit...")
    finally:
        plot_history(history, logger)


def main() -> None:
    args = parse_args()
    logger = setup_logger(Path("train.log"))

    _, confusion_dir, best_model_path, latest_model_path = ensure_checkpoint_dirs(
        args.checkpoint_dir
    )

    model, device, use_autocast, scaler = configure_model_and_device(
        args.model, args.num_classes, logger
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True
    )

    history = _initialise_history()
    best_val_top1 = 0.0
    best_val_loss = float("inf")
    start_epoch = 0
    epochs_without_improvement = 0

    (
        start_epoch,
        best_val_top1,
        best_val_loss,
        epochs_without_improvement,
        history,
    ) = resume_training_if_requested(
        args.resume,
        model,
        optimizer,
        scaler,
        device,
        history,
        logger,
        best_val_top1=best_val_top1,
        best_val_loss=best_val_loss,
        start_epoch=start_epoch,
        epochs_without_improvement=epochs_without_improvement,
    )

    run_training(
        args,
        model,
        optimizer,
        loss_fn,
        device,
        history,
        use_autocast=use_autocast,
        scaler=scaler,
        best_val_top1=best_val_top1,
        best_val_loss=best_val_loss,
        start_epoch=start_epoch,
        epochs_without_improvement=epochs_without_improvement,
        best_model_path=best_model_path,
        latest_model_path=latest_model_path,
        confusion_dir=confusion_dir,
        logger=logger,
    )


if __name__ == "__main__":
    main()
