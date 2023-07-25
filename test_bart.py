import torch

# from transformer.alibi.alibi_lightining import AlibiDecoderLightning
from transformer.decoder.myGPTL import MyGPTLightning
from transformer.HBART.lightning_model import HBARTLMLightning
from data.tokenizer import MyTokenizer


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = MyTokenizer(tokenizer_path="artifacts/general_tokenizer_50")

    modelo = HBARTLMLightning.load_from_checkpoint(
        "modelos_treinados/BART/general/hbart_recon_8_vanillacross_big/last-v1.ckpt"
    )
    modelo.to(device)
    modelo.half()
    modelo.eval()

    conditioned_text = """Finja que você é um gerente de projeto de uma empresa de construção. Descreva um momento em que você teve que tomar uma decisão difícil.

Eu tive que tomar uma decisão difícil quando eu estava trabalhando como gerente de projeto em uma empresa de construção. Eu estava no comando de um projeto que precisava ser concluído até uma certa data, a fim de atender às expectativas do cliente. No entanto, devido a atrasos inesperados, não fomos capazes de cumprir o prazo e, portanto, tive que tomar uma decisão difícil. Decidi estender o prazo, mas tive que garantir que a decisão da equipe fosse ainda mais longa e aumentar o orçamento. Embora fosse possível."""
    decoder_text = """Finja que você é um gerente de projeto de uma empresa de construção. Descreva um momento em que você teve que tomar uma decisão difícil."""

    for _ in range(3):
        gerado = modelo.generate_text(
            cross_attention_text=conditioned_text,
            decoder_text=decoder_text,
            tokenizer=tokenizer,
            max_tokens=100,
            temperature=0.8,
            # p=0.9,
            num_beams=10,
            top_k=30,
            repetition_penalty=1.2,
            device=device,
        )
        print("")
        print(gerado)
