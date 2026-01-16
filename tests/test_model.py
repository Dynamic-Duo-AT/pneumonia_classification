from pneumonia.model import Model

def test_model_initialization():
    """Test the Model class initialization."""
    model = Model()
    assert isinstance(model, Model)

def test_model_forward_pass():
    """Test the forward pass of the Model class."""
    import torch

    model = Model()
    dummy_input = torch.randn(1, 1, 384, 384)  
    output = model(dummy_input)
    assert output.shape == torch.Size([1])
