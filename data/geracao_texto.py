from pathlib import Path
import pandas as pd
import numpy as np

df = pd.read_csv(r"D:\DATASETS\NLP\marketplace_sentiment_analysis_pt\concatenated.csv")

df = df[~df.polarity.isna()].reset_index(drop=True)
df = df.drop_duplicates(["review_text"])
df = df.reset_index(drop=True)
df_filtered = df[df.review_text.str.len() >= 10].reset_index(drop=True)
df_filtered = df_filtered[
    df_filtered.dataset.isin(["b2w", "buscape", "olist"])
].reset_index(drop=True)

pasta_treino = Path(
    r"D:\DATASETS\NLP\marketplace_sentiment_analysis_pt\texts_marketplace\train"
)
pasta_treino.mkdir(parents=True, exist_ok=True)
pasta_validacao = Path(
    r"D:\DATASETS\NLP\marketplace_sentiment_analysis_pt\texts_marketplace\valid"
)
pasta_validacao.mkdir(parents=True, exist_ok=True)
pasta_teste = Path(
    r"D:\DATASETS\NLP\marketplace_sentiment_analysis_pt\texts_marketplace\test"
)
pasta_teste.mkdir(parents=True, exist_ok=True)

target_teste = 10000
target_valid = 10000
target_treino = len(df) - target_teste - target_valid

contagem_teste = 0
contagem_valid = 0
contagem_treino = 0

textos = df_filtered.review_text.tolist()
polaritys = df_filtered.polarity.tolist()

while len(textos) > 0:
    tamanho = len(textos)
    prob_teste = max((target_teste - contagem_teste) / tamanho, 0)
    prob_valid = max((target_valid - contagem_valid) / tamanho, 0)
    prob_treino = max((target_treino - contagem_treino) / tamanho, 0)

    sorteio = np.random.random()
    if sorteio <= prob_teste:
        pasta_destino = pasta_teste
        contagem_teste += 1
    elif sorteio <= prob_teste + prob_valid:
        pasta_destino = pasta_validacao
        contagem_valid += 1
    else:
        pasta_destino = pasta_treino
        contagem_treino += 1

    texto = textos.pop()
    pol = polaritys.pop()

    if pol == 1:
        pasta_destino = pasta_destino / "nota_alta"
    else:
        pasta_destino = pasta_destino / "nota_baixa"

    pasta_destino.mkdir(parents=True, exist_ok=True)

    arquivo_destino = (
        pasta_destino
        / f"texto_avaliacao_{contagem_teste + contagem_treino + contagem_valid}.txt"
    )
    arquivo_destino.write_text(str(texto), encoding="utf8")
