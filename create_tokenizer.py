from pathlib import Path

from data.tokenizer import MyTokenizer

data_paths = [
    "/media/luischary/BIGGER/DATASETS/NLP/AlpacaPT/textos",
    "/media/luischary/BIGGER/DATASETS/NLP/Livros/textos_extraidos",
    "/media/luischary/BIGGER/DATASETS/NLP/OUTROS/corpus_readability_nlp_portuguese-master/1_Ensino_Fundamental_I",
    "/media/luischary/BIGGER/DATASETS/NLP/OUTROS/corpus_readability_nlp_portuguese-master/2_Ensino_Fundamental_II",
    "/media/luischary/BIGGER/DATASETS/NLP/OUTROS/corpus_readability_nlp_portuguese-master/3_Ensino_Medio",
    "/media/luischary/BIGGER/DATASETS/NLP/OUTROS/corpus_readability_nlp_portuguese-master/4_Ensino_Superior",
    "/media/luischary/BIGGER/DATASETS/NLP/OUTROS/enem_2022/textos",
    "/media/luischary/BIGGER/DATASETS/NLP/QA/br_quad_2/textos_train",
    # "/media/luischary/BIGGER/DATASETS/NLP/QA/qa_pt/textos",
    "/media/luischary/BIGGER/DATASETS/NLP/QA/squad-pt/textos_train",
    "/media/luischary/BIGGER/DATASETS/NLP/Summarization/BrWac2Wiki/textos",
    "/media/luischary/BIGGER/DATASETS/NLP/Summarization/CSTNews 6.0/textos",
    "/media/luischary/BIGGER/DATASETS/NLP/Summarization/OpiSums-PT/textos",
    "/media/luischary/BIGGER/DATASETS/NLP/Summarization/portuguese_XLSum_v2/textos_train",
    "/media/luischary/BIGGER/DATASETS/NLP/Summarization/TeMario-ULTIMA VERSAO out2004/textos",
    "/media/luischary/BIGGER/DATASETS/NLP/TJSP/textos",
    "/media/luischary/BIGGER/DATASETS/NLP/Traducao/textos_capes",
    "/media/luischary/BIGGER/DATASETS/NLP/Wikipedia/textos_limpos_wiki_3",
]


def iterate_over_texts():
    for datafolder in data_paths:
        input_path = Path(datafolder)
        for arquivo_de_texto in input_path.glob("*.txt"):
            yield arquivo_de_texto.read_text(encoding="utf8")


if __name__ == "__main__":
    novo_tokenizer = MyTokenizer(
        vocab_size=50000, tokenizer_path="artifacts/general_tokenizer_50"
    )
    novo_tokenizer.train(iterate_over_texts())
