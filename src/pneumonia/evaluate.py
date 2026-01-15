from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from pneumonia.data import create_dataloaders
from pneumonia.model import Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Hydra config setup (same as train)
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "experiments"


def evaluate(model_checkpoint: str, model_type: str, data_dir: str) -> None:
    """
    Evaluate a trained model.

    Args:
        model_checkpoint: Path to the model checkpoint.
        model_type: Type of the model (e.g., 'baseline').
        data_dir: Directory containing the data.

    Returns:
        Accuracy on the test set.
    """
    print("Evaluating...")
    print("Checkpoint:", model_checkpoint)
    print("Model type:", model_type)
    print("Data dir:", data_dir)

    # Load model
    if model_type != "baseline":
        print(f"Using model type: {model_type}")
        model = Model().to(DEVICE)
        model.load_state_dict(torch.load(model_checkpoint))
    else:
        raise NotImplementedError("Only 'baseline' model type is supported in this script.")

    # Create dataloaders
    data_loaders = create_dataloaders(data_dir, batch_size=32)
    test_dataloader = data_loaders["test"]

    model.eval()
    correct, total = 0, 0
    # ensure no gradients are computed during evaluation
    with torch.no_grad():
        # iterate over test data
        for img, target in test_dataloader:
            img, target = img.to(DEVICE), target.to(DEVICE).float()
            y_pred = model(img)
            y_pred_binary = (torch.sigmoid(y_pred) > 0.5).float()
            correct += (y_pred_binary == target).float().sum().item()
            total += target.size(0)

    # compute accuracy
    accuracy = correct / total
    print(f"Test accuracy: {accuracy}")

    return accuracy


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="exp1")
def main(cfg: DictConfig) -> None:
    """
    Main function to run evaluation using Hydra config.
    Args:
        cfg: Hydra configuration object.
    """
    print("Config:\n", OmegaConf.to_yaml(cfg))

    # reuse the SAME config fields you used in training
    evaluate(
        model_checkpoint=cfg.trainer.model_path,
        model_type=cfg.model.name,
        data_dir=cfg.data.path,
        batch_size=cfg.trainer.batch_size,
    )


if __name__ == "__main__":
    main()
