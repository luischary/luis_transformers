from typing import List
from pathlib import Path
import pickle

from tqdm import tqdm

from tokenizer import MyTokenizer


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


def my_text_iterator(data_path: str, limit=1000000):
    contagem = 0
    for arquivo_txt in Path(data_path).glob("**/*.txt"):
        yield arquivo_txt.read_text(encoding="utf8")
        contagem += 1
        if contagem >= limit:
            return


if __name__ == "__main__":
    tokenizer = MyTokenizer(
        vocab_size=60000,
        tokenizer_path="/home/luischary/projetos/transformers_luis/artifacts/marketplace_tokenizer",
    )
    # tokenizer.train(
    #     text_iterator=my_text_iterator(
    #         r"D:\DATASETS\NLP\Wikipedia\textos_limpos_wiki_2"
    #     )
    # )
    create_tokenized_files(
        text_folders=[
            "/media/luischary/BIGGER/DATASETS/NLP/marketplace_sentiment_analysis_pt/texts_marketplace"
        ],
        tokenizer=tokenizer,
        output_path="/home/luischary/projetos/transformers_luis/data/tokenized_datasets/tokens_marketplace_512",
        max_doc_token_len=512,
        min_doc_token_len=50,
        sos_token=1,
        eos_token=2,
    )

    # arquivo = "wiki_1024/tokens_100.pkl"
    # with open(arquivo, "rb") as file:
    #     tokens = pickle.load(file)
    #     print(tokenizer.untokenize_tokens(tokens))
