import shutil

import torch
from pneumonia.data import XRayDataset
from torch.utils.data import Dataset

from tests import _PATH_DATA


def test_xray_dataset():
    """Test the XRayDataset class."""
    dataset = XRayDataset(_PATH_DATA, pre_process_overrule=True)
    assert isinstance(dataset, Dataset)
    assert dataset.__len__() == 0
    assert dataset._is_split_preprocessed("train") is False
    dataset.processed_path.mkdir(parents=True, exist_ok=True)

    train_files = dataset._list_jpegs(dataset.unprocessed_path / "train")
    assert len(train_files) == 4
    mean, std = dataset._compute_mean_std(train_files)
    assert isinstance(mean, torch.Tensor)
    assert isinstance(std, torch.Tensor)

    # makedir processed/train
    (dataset.processed_path / "train").mkdir(parents=True, exist_ok=True)

    dataset._process_and_save_split(
        files=train_files,
        output_dir=dataset.processed_path / "train",
        mean=mean,
        std=std,
    )

    dataset.files = sorted((dataset.processed_path / "train").glob("*.pt"))
    assert dataset.__len__() == 4
    x, label = dataset.__getitem__(0)
    assert isinstance(x, torch.Tensor)
    assert x.shape[0] == 1  # single channel
    assert x.shape[1] == 384  # height
    assert x.shape[2] == 384  # width
    # min and max of normalized tensor
    assert label in (0, 1)

    # Clean up
    shutil.rmtree(dataset.processed_path)
