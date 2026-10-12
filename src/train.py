import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import os
import matplotlib.pyplot as plt

import config
from model import BreastCancerClassifier
from data_loader import get_data_loaders


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None,
                num_epochs=25, device='cuda', patience=5):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 40)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'  {phase:5s} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

                # Check for improvement
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    print(f'  ✅ New best val accuracy: {best_acc:.4f}')
                else:
                    epochs_no_improve += 1

        # Step the scheduler after each epoch
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f'  📉 Learning Rate: {current_lr:.6f}')

        # Early stopping check
        if epochs_no_improve >= patience:
            print(f'\n⏹️ Early stopping triggered after {epoch + 1} epochs (no improvement for {patience} epochs)')
            break

        print()

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history


def main():
    # 1. Data Setup
    print("=" * 50)
    print("Initializing Data Loaders...")
    print("=" * 50)
    train_loader, val_loader, test_loader = get_data_loaders()

    if train_loader is None:
        print("Failed to load data. Exiting.")
        return

    # 2. Model Setup
    print(f"\nInitializing Model (Device: {config.DEVICE})...")
    model = BreastCancerClassifier(model_name='resnet50', num_classes=2, freeze_backbone=True)
    model = model.to(config.DEVICE)

    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # 3. Training Setup
    criterion = nn.CrossEntropyLoss()
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=config.LEARNING_RATE, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=config.LR_STEP_SIZE,
        gamma=config.LR_GAMMA
    )

    # 4. Train
    print("\n" + "=" * 50)
    print("Starting Training...")
    print("=" * 50)
    model, t_loss, v_loss, t_acc, v_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        scheduler=scheduler,
        num_epochs=config.NUM_EPOCHS,
        device=config.DEVICE,
        patience=config.EARLY_STOPPING_PATIENCE
    )

    # 5. Save Model
    save_path = os.path.join(config.MODELS_DIR, 'best_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"\n💾 Model saved to {save_path}")

    # 6. Evaluate on Test Set
    print("\n" + "=" * 50)
    print("Evaluating on Test Set...")
    print("=" * 50)
    model.eval()
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / len(test_loader.dataset)
    print(f"🎯 Test Set Accuracy: {test_acc:.4f}")

    # 7. Plot Metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(t_loss, label='Train Loss', marker='o', markersize=3)
    axes[0].plot(v_loss, label='Val Loss', marker='s', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_acc, label='Train Acc', marker='o', markersize=3)
    axes[1].plot(v_acc, label='Val Acc', marker='s', markersize=3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy History')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(config.MODELS_DIR, 'training_metrics.png')
    plt.savefig(plot_path, dpi=150)
    print(f"📊 Metrics plot saved to {plot_path}")


if __name__ == "__main__":
    main()
