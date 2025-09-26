from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from alexnet import AlexNet
from dataset_imagenett import train_loader, val_loader


def train_loop(dataloader, model, loss_fn, optimizer, device):
    """Run a training epoch and return aggregated loss and accuracy."""

    size = len(dataloader.dataset)
    model.train()

    running_loss = 0.0
    running_corrects = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

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
            print(
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


def evaluate(dataloader, model, loss_fn, device):
    """Evaluate the model and return loss and accuracy."""

    size = len(dataloader.dataset)
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            running_loss += loss.item() * X.size(0)
            running_corrects += (pred.argmax(1) == y).sum().item()

    epoch_loss = running_loss / size
    epoch_acc = running_corrects / size

    return epoch_loss, epoch_acc

if __name__ == "__main__":
    # ---- 1. Device selection – works on CUDA, MPS (Apple Silicon) or CPU ---- 
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using {device} device")
    # Define the model and optimizer
    model = AlexNet(num_classes=1000).to(device)
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

    epochs = 10

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "best_model.pth"
    latest_model_path = checkpoint_dir / "latest_model.pth"

    for t in range(epochs):
        print(f"Epoch {t + 1}/{epochs}\n-------------------------------")
        train_loss, train_acc = train_loop(train_loader, model, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(val_loader, model, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        torch.save(model.state_dict(), latest_model_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with val_acc={val_acc * 100:.2f}%")
        else:
            print(f"Best val_acc so far: {best_val_acc * 100:.2f}%")

        print(
            "Epoch Summary: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
            "val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%".format(
                train_loss=train_loss,
                train_acc=train_acc * 100,
                val_loss=val_loss,
                val_acc=val_acc * 100,
            )
        )

    print("Training complete!")

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

    print(f"Training metrics plot saved to {plot_path.resolve()}")


