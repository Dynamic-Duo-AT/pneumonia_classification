from pneumonia.data import XRayDataset
from torch.utils.data import Dataset


def test_xray_dataset():
    """Test the XRayDataset class."""
    dataset = XRayDataset("data", pre_process_overrule=True)
    assert isinstance(dataset, Dataset)
