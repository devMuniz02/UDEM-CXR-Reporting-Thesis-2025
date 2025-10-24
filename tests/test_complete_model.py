import torch
import torch.nn as nn
from transformers import GPT2Tokenizer

from utils.models.complete_model import (
    DINOEncoder,
    DinoUNet,
    create_decoder,
    LinearProjection,
    CustomModel,
    create_complete_model,
    load_complete_model,
    save_complete_model,
)

from utils.models.modifiedGPT2 import GPT2LMHeadModelModified

def test_create_complete_model():
    model = create_complete_model(
        device="cpu",
        ENCODER_MODEL_PATH=None,
        SEGMENTER_MODEL_PATH=None,
        DECODER_MODEL_PATH=None,
        LINEAR_PROJECTION_PATH=None,
    )
    assert isinstance(model, CustomModel), "Model is not an instance of CustomModel"
    assert isinstance(model.encoder, DINOEncoder), "Encoder is not an instance of DINOEncoder"
    assert isinstance(model.segmenter, DinoUNet), "Segmenter is not an instance of DinoUNet"
    assert isinstance(model.decoder, nn.Module), "Decoder is not an instance of nn.Module"
    assert isinstance(model.decoder, GPT2LMHeadModelModified), "Decoder is not created properly"
    assert isinstance(model.linear_projection, LinearProjection), "Linear projection is not an instance of LinearProjection"
    assert isinstance(model.tokenizer, GPT2Tokenizer), "Tokenizer is not an instance of GPT2Tokenizer"

def test_save_and_load_complete_model(tmp_path):
    model = create_complete_model(
        device="cpu",
        ENCODER_MODEL_PATH=None,
        SEGMENTER_MODEL_PATH=None,
        DECODER_MODEL_PATH=None,
        LINEAR_PROJECTION_PATH=None,
    )
    save_path = tmp_path / "complete_model.pth"
    save_complete_model(model, str(save_path), device="cpu")
    
    loaded_model = load_complete_model(model, str(save_path), device="cpu")
    
    for param_original, param_loaded in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(param_original, param_loaded), "Model parameters do not match after loading"

def test_model_forward_pass():
    model = create_complete_model(
        device="cpu",
        ENCODER_MODEL_PATH=None,
        SEGMENTER_MODEL_PATH=None,
        DECODER_MODEL_PATH=None,
        LINEAR_PROJECTION_PATH=None,
    )
    model.eval()
    
    # Create a dummy input tensor (batch_size=2, channels=3, height=512, width=512)
    dummy_input = torch.randn(2, 3, 512, 512)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    assert hasattr(output, 'logits'), "Output does not have 'logits' attribute"
    assert output.logits.shape[0] == 2, "Output batch size does not match input batch size"
    assert output.logits.shape == (2, 1024, 50257), "Unexpected output shape"