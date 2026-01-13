from glob import glob
from pathlib import Path

import torch
import typer
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor


class XRayDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path, split: str = "train", image_size: int = 384) -> None:
        self.unprocessed_path = Path(data_path) / "raw"
        self.processed_path = Path(data_path) / "processed"
        self.split = split
        self.image_size = image_size

        # Ensure preprocessing has happened for the split
        if not self._is_split_preprocessed(self.split):
            self.preprocess_data()

        self.files = sorted((self.processed_path / self.split).glob("*.pt"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        img_path = self.files[idx]
        x = torch.load(img_path).unsqueeze(0).float() / 255.0  # (1,H,W), [0,1]
        label = 1 if "bacteria" in img_path.name or "virus" in img_path.name else 0
        return x, label

    # ----------------------------
    # Preprocessing orchestration
    # ----------------------------
    def preprocess_data(self):
        self.processed_path.mkdir(parents=True, exist_ok=True)

        train_files = self._list_jpegs(self.unprocessed_path / "train")
        if len(train_files) == 0:
            raise FileNotFoundError(f"No training .jpeg files found under: {self.unprocessed_path / 'train'}")

        print(f"Found {len(train_files)} training images.")

        mean, std = self._compute_mean_std(train_files)

        print("Grayscale mean:", float(mean))
        print("Grayscale std :", float(std))

        for split in ("train", "test", "val"):
            split_files = self._list_jpegs(self.unprocessed_path / split)
            self._process_and_save_split(
                files=split_files,
                output_dir=self.processed_path / split,
                mean=mean,
                std=std,
            )
            print(f"Preprocessing completed {split} ({len(split_files)} images).")

    # ----------------------------
    # Helpers
    # ----------------------------
    def _is_split_preprocessed(self, split: str) -> bool:
        split_dir = self.processed_path / split
        return split_dir.exists() and any(split_dir.glob("*.pt"))

    @staticmethod
    def _list_jpegs(folder: Path) -> list[str]:
        # Recursive glob for .jpeg
        return glob(str(Path(folder) / "**" / "*.jpeg"), recursive=True)

    def _load_and_resize_grayscale(self, file: str) -> Image.Image:
        return Image.open(file).convert("L").resize((self.image_size, self.image_size))

    @staticmethod
    def _normalize_to_uint8(img: Image.Image, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        Convert PIL grayscale image -> normalized tensor -> uint8 tensor for saving.
        Output shape: (H, W) uint8
        """
        x = pil_to_tensor(img).float() / 255.0  # (1,H,W)
        x = (x - mean) / std  # normalized
        x = (x * 255).clamp(0, 255).byte().squeeze(0)  # (H,W) uint8
        return x

    def _compute_mean_std(self, train_files: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        sum_ = 0.0
        sum_sq = 0.0
        num_pixels = 0

        for file in train_files:
            img = self._load_and_resize_grayscale(file)
            x = pil_to_tensor(img).float() / 255.0  # (1,H,W)
            x = x.squeeze(0)  # (H,W)

            sum_ += x.sum()
            sum_sq += (x**2).sum()
            num_pixels += x.numel()

        mean = sum_ / num_pixels
        std = torch.sqrt((sum_sq / num_pixels) - mean**2)
        return mean, std

    def _process_and_save_split(
        self,
        files: list[str],
        output_dir: Path,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            img = self._load_and_resize_grayscale(file)
            x_normalized = self._normalize_to_uint8(img, mean, std)

            # Save as .pt tensor file instead of image
            torch.save(x_normalized, output_dir / f"{Path(file).stem}.pt")


def create_dataloaders(
    data_path: Path, batch_size: int = typer.Option(32, help="Batch size for DataLoaders")
) -> dict[str, DataLoader]:
    """
    Create DataLoaders for train, test, and val splits.

    Args:
        data_path: Path to the data directory.
        batch_size: Batch size for the DataLoaders.

    Returns:
        Dictionary with keys 'train', 'test', 'val' and DataLoader values.
    """
    dataloaders = {}
    for split in ["train", "test", "val"]:
        dataset = XRayDataset(data_path, split)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        dataloaders[split] = dataloader
    return dataloaders


if __name__ == "__main__":
    typer.run(create_dataloaders)
