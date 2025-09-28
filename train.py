from pathlib import Path
import argparse
import logging

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from alexnet import AlexNet
from resnet import resnet18
from dataset_imagenett import train_loader, val_loader


def train_loop(
    dataloader,
    model,
    loss_fn,
    optimizer,
    device,
    *,
    scaler: "torch.cuda.amp.GradScaler | None" = None,
    use_autocast: bool = False,
):
    """Run a training epoch and return aggregated loss and accuracy."""

    size = len(dataloader.dataset)
    model.train()

    running_loss = 0.0
    running_corrects = 0

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
        batch_corrects = (pred.argmax(1) == y).sum().item()

        running_loss += batch_loss * X.size(0)
        running_corrects += batch_corrects

        if batch % 100 == 0:
            current_samples = batch * dataloader.batch_size + len(X)
            avg_loss = running_loss / current_samples
            avg_acc = running_corrects / current_samples
            batch_acc = batch_corrects / len(X)
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "Batch {batch:>4d}: loss={loss:.4f}, acc={acc:.2f}%, "
                "avg_loss={avg_loss:.4f}, avg_acc={avg_acc:.2f}%, lr={lr:.6f} "
                "[{current}/{total}]".format(
                    batch=batch,
                    loss=batch_loss,
                    acc=batch_acc * 100,
                    avg_loss=avg_loss,
                    avg_acc=avg_acc * 100,
                    lr=lr,
                    current=current_samples,
                    total=size,
                )
            )

    epoch_loss = running_loss / size
    epoch_acc = running_corrects / size

    return epoch_loss, epoch_acc


def evaluate(dataloader, model, loss_fn, device, *, use_autocast: bool = False):
    """Evaluate the model and return loss and accuracy."""

    size = len(dataloader.dataset)
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=use_autocast):
                pred = model(X)
                loss = loss_fn(pred, y)

            running_loss += loss.item() * X.size(0)
            running_corrects += (pred.argmax(1) == y).sum().item()

    epoch_loss = running_loss / size
    epoch_acc = running_corrects / size

    return epoch_loss, epoch_acc

def build_model(name: str, num_classes: int) -> nn.Module:
    """Factory method to construct a supported classification model."""

    name = name.lower()
    if name == "alexnet":
        return AlexNet(num_classes=num_classes)
    if name in {"resnet", "resnet18"}:
        return resnet18(num_classes=num_classes)

    raise ValueError(
        f"Unsupported model '{name}'. Available options: alexnet, resnet18"
    )


if __name__ == "__main__":
    # Create and configure logger
    logging.basicConfig(filename="train.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
    # Creating an object
    logger = logging.getLogger()

    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)

    # Arguments for the script
    parser = argparse.ArgumentParser(description="Train image classification models")
    parser.add_argument(
        "--model",
        default="alexnet",
        help="Model architecture to train (alexnet or resnet18)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Number of output classes",
    )
    args = parser.parse_args()

    # ---- 1. Device selection – works on CUDA, MPS (Apple Silicon) or CPU ----
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    logger.info(f"Using {device} device")
    logger.info(f"Selected model: {args.model}")

    # Define the model and optimizer
    model = build_model(args.model, args.num_classes).to(device)

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_autocast = False
    scaler = None
    if device.type == "cuda":
        logger.info(f"Detected {gpu_count} CUDA device(s)")
        device_capabilities = []
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
    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()
    # Initialize the optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,                     # initial learning rate
        momentum=0.9,                # classic momentum
        weight_decay=5e-4,           # L2 regularisation
        nesterov=True                # optional
    )

    epochs = 100

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "best_model.pth"
    latest_model_path = checkpoint_dir / "latest_model.pth"

    for t in range(epochs):
        logger.info(f"Epoch {t + 1}/{epochs}\n-------------------------------")
        train_loss, train_acc = train_loop(
            train_loader,
            model,
            loss_fn,
            optimizer,
            device,
            scaler=scaler,
            use_autocast=use_autocast,
        )
        val_loss, val_acc = evaluate(
            val_loader,
            model,
            loss_fn,
            device,
            use_autocast=use_autocast,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        state_dict = (
            model.module.state_dict()
            if isinstance(model, nn.DataParallel)
            else model.state_dict()
        )

        torch.save(state_dict, latest_model_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(state_dict, best_model_path)
            logger.info(f"New best model saved with val_acc={val_acc * 100:.2f}%")
        else:
            logger.info(f"Best val_acc so far: {best_val_acc * 100:.2f}%")

        logger.info(
            "Epoch Summary: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
            "val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%".format(
                train_loss=train_loss,
                train_acc=train_acc * 100,
                val_loss=val_loss,
                val_acc=val_acc * 100,
            )
        )

    logger.info("Training complete!")

    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, [acc * 100 for acc in history["train_acc"]], label="Train Acc")
    plt.plot(epochs_range, [acc * 100 for acc in history["val_acc"]], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy over Epochs")
    plt.legend()

    plot_path = Path("training_metrics.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    logger.info(f"Training metrics plot saved to {plot_path.resolve()}")


