import torch
import numpy as np

from data.datasets import VariableLenEncoderMLMDataset


# test
if __name__ == "__main__":
    meu_dataset = VariableLenEncoderMLMDataset(
        tokens_metadata="data/infos_tokens.pq",
        max_len=16384,
        min_len=512,
        batches={500: 32, 1000: 16, 2000: 8, 4000: 4, 8000: 2, 16000: 1},
        limit=10000,
        begin_sentence_token=1,
        end_of_sentence_token=2,
        pad_token_id=0,
        segment_size=32,
        mask_prob=0.15,
        special_tokens=[0, 1, 2, 3, 4, 5, 6, 7],
        mask_token_id=3,
        vocab_size=50000,
    )

    # print(len(meu_dataset))
    # resposta = meu_dataset[len(meu_dataset)]
    # for r in resposta:
    #     print(r.shape)
    for i in range(len(meu_dataset) + 1):
        masked_tensors, tokenized_tensors, mask_tensors = meu_dataset[i]
        print(masked_tensors.shape)
        # print(tokenized_tensors.shape)
        # print(mask_tensors.shape)

    print(meu_dataset.tokens_dataset)
