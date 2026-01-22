from pathlib import Path

import hydra
import torch
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from pneumonia.data import create_dataloaders
from pneumonia.model import Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Hydra config setup (same as train)
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "experiments"


def evaluate(model_checkpoint: str, model_type: str, data_dir: str, num_workers: int, batch_size: int = 32, contrast: bool = False) -> None:
    """
    Evaluate a trained model.

    Args:
        model_checkpoint: Path to the model checkpoint.
        model_type: Type of the model (e.g., 'baseline').
        data_dir: Directory containing the data.
        num_workers: Number of worker threads for data loading.
        batch_size: Batch size for evaluation.
        contrast: Whether to use contrast images.
    Returns:
        Accuracy on the test set.
    """
    # Logging
    logger.info("Evaluating...")
    logger.info(f"Checkpoint: {model_checkpoint}")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Number of workers: {num_workers}")
    logger.info(f"Batch size: {batch_size}")

    # Load model
    if model_type == "baseline":
        logger.info(f"Using model type: {model_type}")
        model = Model().to(DEVICE)
        model.load_state_dict(torch.load(model_checkpoint))
    else:
        raise NotImplementedError("Only 'baseline' model type is supported in this script.")

    # Create dataloaders
    data_loaders = create_dataloaders(data_dir, num_workers=num_workers, batch_size=batch_size)
    test_dataloader = data_loaders["test"]

    model.eval()
    correct, total = 0, 0
    # ensure no gradients are computed during evaluation
    with torch.no_grad():
        # iterate over test data
        for img, target in test_dataloader:
            img, target = img.to(DEVICE), target.to(DEVICE).float()
            if contrast:
                logger.info("Using contrast images for evaluation")
                # increase contrast in the normalized image
                img = img * 3.5  

            # forward pass    
            y_pred = model(img)
            y_pred_binary = (torch.sigmoid(y_pred) > 0.5).float()
            correct += (y_pred_binary == target).float().sum().item()
            total += target.size(0)

    # compute accuracy
    accuracy = correct / total

    # log
    logger.info(f"Test accuracy: {accuracy}")
    wandb.log({"test/accuracy": accuracy})

    # so you can view without looking at logs
    print(f"Test accuracy: {accuracy}")

    return accuracy


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="exp1")
def main(cfg: DictConfig) -> None:
    """
    Main function to run evaluation using Hydra config.
    Args:
        cfg: Hydra configuration object.
    """
    # setup logger
    logger.remove()
    logger.add(
        cfg.loguru.log_dir + "/test.log",
        level=cfg.loguru.level,
        format="{time} {level} {message}",
    )
    logger.info("Config:\n", OmegaConf.to_yaml(cfg))

    # check if wandb_run_id.txt exists to resume the same run
    path = Path("wandb_run_id.txt")
    if path.exists():
        with open(path, "r") as f:
            run_id = f.read().strip()
        wandb.init(project="pneumonia-classification", id=run_id, resume="must")
    else:
        wandb.init(project="pneumonia-classification")

    # reuse the SAME config fields you used in training
    evaluate(
        model_checkpoint=cfg.trainer.model_path,
        model_type=cfg.model.name,
        data_dir=cfg.data.path,
        num_workers=cfg.trainer.num_workers,
        batch_size=cfg.trainer.batch_size,
        contrast=bool(cfg.evaluate.contrast)
    )


if __name__ == "__main__":
    main()
