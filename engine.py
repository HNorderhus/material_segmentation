import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from utils import save_model, plt_to_tensor, initialize_metrics, calculate_metrics, update_running_means, \
    visualize_confusion_matrix


def train_step(model, dataloader, loss_fn, optimizer, device, metrics):
    model.train()
    running_loss = 0.0
    running_iou_means, running_ltiou_means = [], []

    for i_batch, sample_batched in enumerate(dataloader):
        optimizer.zero_grad()
        inputs, labels = sample_batched
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)["out"]
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        loss.backward()
        optimizer.step()

        iou_values, lt_iou = calculate_metrics(preds, labels, metrics)
        running_iou_means.append(iou_values)
        running_ltiou_means = update_running_means(running_ltiou_means, lt_iou)

    train_acc = torch.mean(torch.stack(running_iou_means), dim=0)
    lt_iou_acc = np.mean(running_ltiou_means) if running_ltiou_means else 0.
    train_loss = running_loss / len(dataloader)

    return train_loss, train_acc, lt_iou_acc, metrics["confmat"].compute().cpu().numpy()


def val_step(model, dataloader, loss_fn, device, metrics):
    model.eval()
    running_iou_means, running_ltiou_means = [], []
    val_loss = 0

    with torch.inference_mode():
        for i_batch, sample_batched in enumerate(dataloader):
            inputs, labels = sample_batched
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)["out"]
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)

            iou_values, lt_iou = calculate_metrics(preds, labels, metrics)
            running_iou_means.append(iou_values)
            running_ltiou_means = update_running_means(running_ltiou_means, lt_iou)

        val_acc = torch.mean(torch.stack(running_iou_means), dim=0)
        # Calculate lt_iou_acc mean outside the function after the loop
        lt_iou_acc = np.mean(running_ltiou_means) if running_ltiou_means else 0.

    final_precision = metrics["precision"].compute()
    final_recall = metrics["recall"].compute()
    final_f1 = metrics["f1_score"].compute()

    val_loss = val_loss / len(dataloader)
    return val_loss, val_acc, lt_iou_acc, metrics[
        "confmat"].compute().cpu().numpy(), final_precision, final_recall, final_f1


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          writer: torch.utils.tensorboard.writer.SummaryWriter,
          name: str,
          patience: int = 30,
          train_pruned: bool = False
          ):
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    # Initialize metrics
    metrics = initialize_metrics(device)

    # Loop through training and testing steps for a number of epochs
    with writer:
        for epoch in tqdm(range(epochs)):
            # Train step
            train_loss, train_iou, train_lt_iou, train_confmat = train_step(model=model,
                                                                            dataloader=train_dataloader,
                                                                            loss_fn=loss_fn,
                                                                            optimizer=optimizer,
                                                                            device=device,
                                                                            metrics=metrics)

            # Validation step
            val_loss, val_iou, val_lt_iou, val_confmat, val_precision, val_recall, val_f1 = val_step(
                model=model,
                dataloader=val_dataloader,
                loss_fn=loss_fn,
                device=device,
                metrics=metrics
            )

            # Print results to console
            print(f"Epoch: {epoch + 1} | "
                  f"train_loss: {train_loss:.4f} | "
                  f"train_iou: {train_iou:.4f} | "
                  f"val_loss: {val_loss:.4f} | "
                  f"val_iou: {val_iou:.4f} | "
                  f"val_lt_iou: {val_lt_iou:.4f} | ")

            # Add results to SummaryWriter
            writer.add_scalars("Loss", {"train_loss": train_loss, "val_loss": val_loss}, global_step=epoch)
            writer.add_scalars("IoU", {"train_iou": train_iou, "val_iou": val_iou}, global_step=epoch)
            writer.add_scalars("Validation/Metrics",
                               {"Precision": val_precision, "Recall": val_recall, "F1_Score": val_f1},
                               global_step=epoch)

            # Check for improvement and save model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # if train_pruned:
                save_model(model=model, target_dir="results/models", model_name=f"{name}_weights.pth")
                # else:
                #     save_model(model=model, target_dir="results/models", model_name=f"{name}.pth")

            else:
                patience_counter += 1
                print(f"Epoch {epoch + 1}: No improvement in validation loss.")

            # Early stopping condition
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                visualize_confusion_matrix(val_confmat, epoch)
                writer.add_image('Val Confusion Matrix', plt_to_tensor(plt), global_step=epoch)
                break
