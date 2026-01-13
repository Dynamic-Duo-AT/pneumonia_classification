from pathlib import Path
import wandb
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from pneumonia.data import create_dataloaders
from pneumonia.model import Model

# Training script. Add description later.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(
    lr: float = 0.001,
    batch_size: int = 32,
    epochs: int = 1,
    model_path: str = "models/model.pth",
    model: str = "baseline",
    data_path: str = "data/",
) -> None:
    print("Training started...")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    wandb.init(
        entity="Dynamic_Duo",
        project="Pneumonia-Classification",
        config={"lr": lr, "batch_size": batch_size, "epochs": epochs},
    )

    # model selection
    if model == "baseline":
        model = Model().to(DEVICE)
    else:
        raise ValueError("Model not implemented.")

    # Creating three dataloaders for train, val and test sets
    dataloaders = create_dataloaders(data_path, batch_size=batch_size)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        preds, targets = [], []
        for i, (img, target) in enumerate(dataloaders["train"]):
            img, target = img.to(DEVICE), target.to(DEVICE).float()
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            preds_binary = (torch.sigmoid(y_pred) > 0.5).float()
            accuracy = (preds_binary == target).float().mean().item()
            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

        # Validation loop
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for img, target in dataloaders["val"]:
                img, target = img.to(DEVICE), target.to(DEVICE).float()
                y_pred = model(img)
                val_loss = loss_fn(y_pred, target)
                val_preds_binary = (torch.sigmoid(y_pred) > 0.5).float()
                val_accuracy = (val_preds_binary == target).float().mean().item()
                wandb.log({"val_loss": val_loss.item(), "val_accuracy": val_accuracy})

                val_preds.append(y_pred.detach().cpu())
                val_targets.append(target.detach().cpu())
    print("Training completed.")

    # Save the trained model
    torch.save(model.state_dict(), model_path)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "experiments"


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="exp1")
def main(cfg: DictConfig) -> None:
    print("Config:\n", OmegaConf.to_yaml(cfg))

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
