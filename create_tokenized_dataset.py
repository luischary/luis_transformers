from typing import List
from pathlib import Path
import pickle

from tqdm import tqdm
import pandas as pd

from data.tokenizer import MyTokenizer

data_paths = [
    # "/media/luischary/BIGGER/DATASETS/NLP/AlpacaPT/textos",
    # "/media/luischary/BIGGER/DATASETS/NLP/Livros/textos_extraidos",
    # "/media/luischary/BIGGER/DATASETS/NLP/OUTROS/corpus_readability_nlp_portuguese-master/1_Ensino_Fundamental_I",
    # "/media/luischary/BIGGER/DATASETS/NLP/OUTROS/corpus_readability_nlp_portuguese-master/2_Ensino_Fundamental_II",
    # "/media/luischary/BIGGER/DATASETS/NLP/OUTROS/corpus_readability_nlp_portuguese-master/3_Ensino_Medio",
    # "/media/luischary/BIGGER/DATASETS/NLP/OUTROS/corpus_readability_nlp_portuguese-master/4_Ensino_Superior",
    # "/media/luischary/BIGGER/DATASETS/NLP/OUTROS/enem_2022/textos",
    # "/media/luischary/BIGGER/DATASETS/NLP/QA/br_quad_2/textos_train",
    # # "/media/luischary/BIGGER/DATASETS/NLP/QA/qa_pt/textos",
    # "/media/luischary/BIGGER/DATASETS/NLP/QA/squad-pt/textos_train",
    # "/media/luischary/BIGGER/DATASETS/NLP/Summarization/BrWac2Wiki/textos",
    # "/media/luischary/BIGGER/DATASETS/NLP/Summarization/CSTNews 6.0/textos",
    # "/media/luischary/BIGGER/DATASETS/NLP/Summarization/OpiSums-PT/textos",
    # "/media/luischary/BIGGER/DATASETS/NLP/Summarization/portuguese_XLSum_v2/textos_train",
    # "/media/luischary/BIGGER/DATASETS/NLP/Summarization/TeMario-ULTIMA VERSAO out2004/textos",
    # "/media/luischary/BIGGER/DATASETS/NLP/TJSP/textos",
    # "/media/luischary/BIGGER/DATASETS/NLP/Traducao/textos_capes",
    # "/media/luischary/BIGGER/DATASETS/NLP/Wikipedia/textos_limpos_wiki_3",
    "/media/luischary/BIGGER/DATASETS/NLP/cc100",
]


def iterate_over_texts():
    for datafolder in data_paths:
        input_path = Path(datafolder)
        for arquivo_de_texto in input_path.glob("*.txt"):
            yield arquivo_de_texto.read_text(encoding="utf8")


def create_tokenized_files(
    text_folders: List[str],
    tokenizer: MyTokenizer,
    output_path: str,
    max_doc_token_len: int = 512,
    min_doc_token_len: int = 50,
    sos_token: int = 1,
    eos_token: int = 2,
):
    out_folder = Path(output_path)
    out_folder.mkdir(parents=True, exist_ok=True)

    doc_count = 0

    all_tokens = []
    for text_folder in text_folders:
        print(f"Tratando textos - {text_folder}")
        for doc in tqdm(Path(text_folder).glob("**/*.txt")):
            doc_tokenized = (
                [sos_token]
                + tokenizer.tokenize_text(doc.read_text(encoding="utf8"))
                + [eos_token]
            )
            if len(doc_tokenized) >= min_doc_token_len:
                all_tokens += doc_tokenized

            while len(all_tokens) >= max_doc_token_len:
                sub_tokens = all_tokens[:max_doc_token_len]
                file_path = out_folder / f"tokens_{doc_count}.pkl"
                with open(file_path, "wb") as file:
                    pickle.dump(sub_tokens, file)
                doc_count += 1
                all_tokens = all_tokens[max_doc_token_len:]

    file_path = out_folder / f"tokens_{doc_count}.pkl"
    with open(file_path, "wb") as file:
        pickle.dump(all_tokens, file)


def tokenize_folders(
    text_folders: List[str],
    tokenizer: MyTokenizer,
    output_path: str,
    sos_token: int = 1,
    eos_token: int = 2,
):
    out_folder = Path(output_path)
    out_folder.mkdir(parents=True, exist_ok=True)

    for folder in text_folders:
        print(f"Tratando textos - {folder}")
        f = Path(folder)
        parent = f.parent.stem[-20:]
        parent_2 = f.parent.parent.stem[-20:]
        parent_3 = f.parent.parent.parent.stem[-20:]

        output_doc_folder = out_folder / f"{parent_3}_{parent_2}_{parent}"
        output_doc_folder.mkdir(parents=True, exist_ok=True)

        doc_id = 1
        for doc in tqdm(Path(folder).glob("*.txt")):
            doc_tokenized = (
                [sos_token]
                + tokenizer.tokenize_text(doc.read_text(encoding="utf8"))
                + [eos_token]
            )
            file_path = output_doc_folder / f"{doc_id}.pkl"
            with open(file_path, "wb") as file:
                pickle.dump(doc_tokenized, file)

            doc_id += 1


def get_tokens_stats(datatokens_folder: str):
    segment_size = 32
    dados = {
        "parent_folder": [],
        "doc_id": [],
        "num_tokens": [],
        # "token_bin": [],
        "tokens_ref": [],
    }
    # token_bins = [(segment_size - 1) * i + 512 for i in range(1026)]
    # token_bins = [128, 256] + token_bins

    for file in tqdm(Path(datatokens_folder).glob("**/*.pkl")):
        with open(file, "rb") as arquivo:
            tokens = pickle.load(arquivo)

        dados["doc_id"].append(file.stem)
        dados["num_tokens"].append(len(tokens))
        dados["parent_folder"].append(file.parent.stem)
        dados["tokens_ref"].append(str(file.relative_to(Path("./"))))

        # bin_index = 0
        # while len(tokens) > token_bins[bin_index]:
        #     bin_index += 1
        #     if bin_index >= len(token_bins):
        #         bin_index = len(token_bins) - 1
        #         break
        # dados["token_bin"].append(token_bins[bin_index])

    df = pd.DataFrame(dados)
    print(df)
    print(df.num_tokens.describe())
    # df.to_csv("data/infos_tokens.csv", sep=";", encoding="utf8")
    df.to_parquet("data/infos_tokens.pq")


if __name__ == "__main__":
    tokenizer = MyTokenizer(
        vocab_size=50000,
        tokenizer_path="artifacts/general_tokenizer_50",
    )
    # tokenizer.train(
    #     text_iterator=my_text_iterator(
    #         r"D:\DATASETS\NLP\Wikipedia\textos_limpos_wiki_2"
    #     )
    # )

    # create_tokenized_files(
    #     text_folders=data_paths,
    #     tokenizer=tokenizer,
    #     output_path="data/tokenized_datasets/general_1024",
    #     max_doc_token_len=1024,
    #     min_doc_token_len=100,
    #     sos_token=1,
    #     eos_token=2,
    # )

    # tokenize_folders(
    #     text_folders=data_paths,
    #     tokenizer=tokenizer,
    #     output_path="data/datatokens",
    #     sos_token=1,
    #     eos_token=2,
    # )

    get_tokens_stats("data/datatokens")
