import torch

from transformer.longformer.longformer_lightining import LongformerMLMLightning
from transformer.vanilla.encoder_lightining import VanillaEncoderMLMLightning
from transformer.hierarchical.hierarchical_lightining import (
    HierarchicalEncoderMLMLightning,
)
from transformer.mlm_generation import MLMPipeline
from data.tokenizer import MyTokenizer


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    # tokenizer = MyTokenizer(tokenizer_path="artifacts/marketplace_tokenizer")
    tokenizer = MyTokenizer(tokenizer_path="artifacts/general_tokenizer_50")

    # parametros_modelo = {
    #     "vocab_size": 60000,
    #     "embed_dim": 256,
    #     "num_layers": 12,
    #     "num_heads": 8,
    #     "hidden_size": 256 * 4,
    #     "dropout": 0.1,
    #     "pad_token_id": 0,
    #     "tokens_per_sample": 128,
    # }
    # modelo = AlibiDecoderLightning.load_from_checkpoint(
    #     "lightning_logs/version_1/checkpoints/epoch=19-step=29260.ckpt",
    #     **parametros_modelo
    # )
    # modelo = AlibiDecoderLightning.load_from_checkpoint(
    #     "modelos_treinados/decoders/marketplace/last-v2.ckpt"
    # )

    encoder_params = {
        "vocab_size": 60000,
        "embed_dim": 512,
        "hidden_size": 512 * 4,
        "num_heads": 12,
        "num_layers": 12,
        "window_size": 32,
        "max_context_size": 512,
        "dropout": 0.1,
    }

    # modelo = LongformerMLMLightning.load_from_checkpoint(
    #     "modelos_treinados/encoders/marketplace/longformer_128_512_marketplace/last.ckpt",
    #     map_location=device,
    # )
    modelo = HierarchicalEncoderMLMLightning.load_from_checkpoint(
        "modelos_treinados/encoders/general/HENCODER_VLEN_8k_500k/last.ckpt",
        map_location=device,
    )

    modelo.to(device)
    modelo.eval()

    input_text = (
        "Dados os pedidos e os autos fornecidos, julgo a ação<mask>a favor do autor."
    )
    pipe = MLMPipeline(
        mask_token_id=3, tokenizer=tokenizer, model=modelo, sos_token=1, eos_token=2
    )
    resposta = pipe.predict_mask(
        input_text, device, top_k=5, pad_input=True, padding=False
    )
    pipe.print_pretty(input_text, *resposta)
