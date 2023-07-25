from pathlib import Path
import pickle
import os

from tqdm import tqdm
from datasketch import MinHash, MinHashLSH


# a pure python shingling function that will be used in comparing
# LSH to true Jaccard similarities
def shingles(text, char_ngram=5):
    return set(
        text[head : head + char_ngram] for head in range(0, len(text) - char_ngram + 1)
    )


def jaccard(set_a, set_b):
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def get_duplicated_files(
    main_folder: Path, num_perm: int = 128, sim_threshold: float = 0.98
):
    nao_duplicados = []
    duplicados = []
    lsh = MinHashLSH(
        threshold=sim_threshold,
        num_perm=num_perm,
    )
    for file_index, file_path in tqdm(enumerate(main_folder.glob("**/*.pkl"))):
        # with open(file_path, "rb") as file:
        #     tokens = pickle.load(file)

        mh = MinHash(num_perm=num_perm)
        # for int_num in tokens:
        #     mh.update(str(int_num).encode())
        # tokens_bin = [str(t).encode() for t in tokens]
        # mh.update_batch(tokens_bin)
        mh.update(file_path.read_bytes())

        # ve se ja nao existe duplicado
        resposta = lsh.query(mh)
        if len(resposta) > 0:
            duplicados.append(str(file_path))
        else:
            # nao eh duplicado
            nao_duplicados.append(str(file_path))
            lsh.insert(str(file_path), mh, check_duplication=False)

        # if file_index >= 100000:
        #     break
    return nao_duplicados, duplicados


if __name__ == "__main__":
    nao_duplicados, duplicados = get_duplicated_files(
        Path("data/datatokens"), num_perm=256, sim_threshold=0.99
    )
    print(f"Quantidade de duplicados: {len(duplicados)}")
    print(f"Quantidade de nao-duplicados: {len(nao_duplicados)}")

    # apagando os duplicados
    for file_path in tqdm(duplicados):
        os.remove(file_path)
