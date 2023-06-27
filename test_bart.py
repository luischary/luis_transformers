import torch

# from transformer.alibi.alibi_lightining import AlibiDecoderLightning
from transformer.decoder.myGPTL import MyGPTLightning
from transformer.HBART.lightning_model import HBARTLMLightning
from data.tokenizer import MyTokenizer


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = MyTokenizer(tokenizer_path="artifacts/general_tokenizer_50")

    modelo = HBARTLMLightning.load_from_checkpoint(
        "modelos_treinados/BART/general/hbart_reconstruction/last-v1.ckpt"
    )
    modelo.to(device)
    modelo.half()
    modelo.eval()

    conditioned_text = "O Brasil é um dos maiores países da América Latina. Ele é muito conhecido por suas festas e comidas típicas como carnaval e feijoada."
    decoder_text = "O Brasil é"

    for _ in range(5):
        gerado = modelo.generate_text(
            cross_attention_text=conditioned_text,
            decoder_text=decoder_text,
            tokenizer=tokenizer,
            max_tokens=50,
            temperature=0.7,
            # p=0.8,
            num_beams=3,
            top_k=3,
            repetition_penalty=1.0,
            device=device,
        )
        print("")
        print(gerado)
