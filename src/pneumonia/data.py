from glob import glob
from pathlib import Path

import torch
import typer
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor


class XRayDataset(Dataset):
    def __init__(
        self, data_path: Path, pre_process_overrule: bool = False, split: str = "train", image_size: int = 384
    ) -> None:
        """
        Dataset for Pneumonia X-Ray images.

        Args:
            data_path: Path to the data directory.
            pre_process_overrule: If True, forces preprocessing even if already done.
            split: One of 'train', 'test', or 'val'.
            image_size: Size to which images are resized (image_size x image_size).
        """
        self.unprocessed_path = Path(data_path) / "raw"
        self.processed_path = Path(data_path) / "processed"
        self.split = split
        self.image_size = image_size

        # Ensure preprocessing has happened for the split
        if not self._is_split_preprocessed(self.split) and pre_process_overrule is False:
            logger.info(f"Preprocessing data for split: {self.split}")
            self.preprocess_data()

        self.files = sorted((self.processed_path / self.split).glob("*.pt"))
        logger.info(f"Loaded {len(self.files)} samples for split: {self.split}")

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return len(self.files)

    def __getitem__(self, idx: int):
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple of (image tensor, label).
        """
        img_path = self.files[idx]
        # load the float tensor saved as .pt
        x = torch.load(img_path)  # (1,H,W) normalized float tensor
        label = 1 if "bacteria" in img_path.name or "virus" in img_path.name else 0
        return x, label

    def preprocess_data(self):
        """
        Preprocess raw images and save as tensors.
        """
        self.processed_path.mkdir(parents=True, exist_ok=True)

        train_files = self._list_jpegs(self.unprocessed_path / "train")

        logger.info(f"Found {len(train_files)} training images.")

        mean, std = self._compute_mean_std(train_files)

        logger.info("Grayscale mean:", float(mean))
        logger.info("Grayscale std :", float(std))

        for split in ("train", "test", "val"):
            split_files = self._list_jpegs(self.unprocessed_path / split)
            self._process_and_save_split(
                files=split_files,
                output_dir=self.processed_path / split,
                mean=mean,
                std=std,
            )
            logger.info(f"Preprocessing completed {split} ({len(split_files)} images).")

    def _is_split_preprocessed(self, split: str) -> bool:
        """
        Check if a split has been preprocessed.

        Args:
            split: One of 'train', 'test', or 'val'.

        Returns:
            True if preprocessed data exists for the split, False otherwise.
        """
        split_dir = self.processed_path / split
        return split_dir.exists() and any(split_dir.glob("*.pt"))

    @staticmethod
    def _list_jpegs(folder: Path) -> list[str]:
        """
        List all .jpeg files in a folder recursively.

        Args:
            folder: Path to the folder.

        Returns:
            List of .jpeg file paths.
        """
        # Recursive glob for .jpeg
        return glob(str(Path(folder) / "**" / "*.jpeg"), recursive=True)

    def _load_and_resize_grayscale(self, file: str) -> Image.Image:
        """ "
        Load an image file, convert to grayscale, and resize.

        Args:
            file: Path to the image file.

        Returns:
            Resized grayscale PIL Image.
        """
        return Image.open(file).convert("L").resize((self.image_size, self.image_size))

    @staticmethod
    def _normalize(img: Image.Image, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        Normalize a PIL image tensor.

        Args:
            img: PIL Image.
            mean: Mean tensor.
            std: Std tensor.

        Returns:
            Normalized image tensor.
        """
        x = pil_to_tensor(img).float() / 255.0  # (1,H,W)
        x = (x - mean) / std  # normalized
        return x

    def _compute_mean_std(self, train_files: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and std of grayscale images in the training set.

        Args:
            train_files: List of training image file paths.

        Returns:
            Tuple of (mean, std) tensors.
        """
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
        """
        Process and save all images in a split.

        Args:
            files: List of image file paths.
            output_dir: Directory to save processed tensors.
            mean: Mean tensor for normalization.
            std: Std tensor for normalization.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            img = self._load_and_resize_grayscale(file)
            x_normalized = self._normalize(img, mean, std)

            # Save as .pt tensor file instead of image
            torch.save(x_normalized, output_dir / f"{Path(file).stem}.pt")


def create_dataloaders(
    data_path: Path, num_workers: int ,batch_size: int = typer.Option(32, help="Batch size for DataLoaders")
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
        dataset = XRayDataset(data_path, False, split)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        dataloaders[split] = dataloader
    return dataloaders


if __name__ == "__main__":
    typer.run(create_dataloaders)
