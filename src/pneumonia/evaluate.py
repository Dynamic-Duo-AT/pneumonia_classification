
import torch
import typer

from pneumonia.data import create_dataloaders
from pneumonia.model import Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating...")
    print(model_checkpoint)

    model = Model().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    data_loaders = create_dataloaders("data/", batch_size=32)
    test_dataloader = data_loaders["test"]

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE).float()
        y_pred = model(img)
        y_pred_binary = (torch.sigmoid(y_pred) > 0.5).float()
        correct += (y_pred_binary == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    typer.run(evaluate)