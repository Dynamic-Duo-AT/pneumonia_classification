from pathlib import Path

import hydra
import torch
from loguru import logger
import wandb
from omegaconf import DictConfig, OmegaConf

from pneumonia.data import create_dataloaders
from pneumonia.model import Model

# Training script. Add description later.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Hydra config setup
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "experiments"


def train(
    lr: float = 0.001,
    batch_size: int = 32,
    epochs: int = 1,
    model_path: str = "models/model.pth",
    model: str = "baseline",
    data_path: str = "data/",
) -> None:
    """
    Train a pneumonia classification model.

    Args:
        lr: Learning rate for the optimizer.
        batch_size: Batch size for training.
        epochs: Number of training epochs.
        model_path: Path to save the trained model.
        model: Model architecture to use.
        data_path: Path to the dataset.
    """
    logger.info("Training started...")
    logger.info(f"{lr=}, {batch_size=}, {epochs=}")

    # Initialize wandb
    wandb.init(
        entity="Dynamic_Duo",
        project="Pneumonia-Classification",
        config={"lr": lr, "batch_size": batch_size, "epochs": epochs},
    )

    # model selection
    if model == "baseline":
        model = Model().to(DEVICE)
        logger.info("Using baseline model.")
    else:
        logger.error(f"Model {model} not implemented.")
        raise ValueError("Model not implemented.")

    # Creating three dataloaders for train, val and test sets
    logger.info("Creating dataloaders...")
    dataloaders = create_dataloaders(data_path, batch_size=batch_size)

    # defining loss function and optimizer
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info("Starting training loop...")
    # Training loop
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(dataloaders["train"]):
            # processing batch
            img, target = img.to(DEVICE), target.to(DEVICE).float()
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            preds_binary = (torch.sigmoid(y_pred) > 0.5).float()
            accuracy = (preds_binary == target).float().mean().item()
            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

            if i % 100 == 0:
                logger.info(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

        # Validation loop
        model.eval()
        with torch.no_grad():
            for img, target in dataloaders["val"]:
                # processing batch
                img, target = img.to(DEVICE), target.to(DEVICE).float()
                y_pred = model(img)
                val_loss = loss_fn(y_pred, target)
                val_preds_binary = (torch.sigmoid(y_pred) > 0.5).float()
                val_accuracy = (val_preds_binary == target).float().mean().item()
                wandb.log({"val_loss": val_loss.item(), "val_accuracy": val_accuracy})
    logger.info("Training completed.")

    # Save the trained model
    logger.info(f"Saving model to {model_path}...")
    torch.save(model.state_dict(), model_path)


# Hydra main function
@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="exp1")
def main(cfg: DictConfig) -> None:
    """
    Main function to run training with Hydra config.

    Args:
        cfg: Hydra configuration object.
    """
    # setup logger
    logger.remove()
    logger.add(
        cfg.loguru.log_dir + "/train.log",
        level=cfg.loguru.level,
        format="{time} {level} {message}",
    )

    logger.info("Config:\n", OmegaConf.to_yaml(cfg))

    train(
        lr=cfg.trainer.lr,
        batch_size=cfg.trainer.batch_size,
        epochs=cfg.trainer.epochs,
        model_path=cfg.trainer.model_path,
        model=cfg.model.name,
        data_path=cfg.data.path,
    )


if __name__ == "__main__":
    main()
