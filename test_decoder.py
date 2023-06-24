import torch

# from transformer.alibi.alibi_lightining import AlibiDecoderLightning
from transformer.decoder.myGPTL import MyGPTLightning
from data.tokenizer import MyTokenizer


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = MyTokenizer(tokenizer_path="artifacts/marketplace_tokenizer")

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
    modelo = MyGPTLightning.load_from_checkpoint(
        "modelos_treinados/decoders/marketplace/mygpt_128_512_2_marketplace/last.ckpt"
    )
    modelo.to(device)
    modelo.half()
    modelo.eval()

    input_text = "O que eu gostei da geladeira:"
    for _ in range(5):
        gerado = modelo.generate_text(
            input_text=input_text,
            tokenizer=tokenizer,
            max_tokens=100,
            temperature=0.8,
            p=0.8,
            # num_beams=3,
            # top_k=20,
            repetition_penalty=1.2,
            device=device,
        )
        print("")
        print(gerado)
