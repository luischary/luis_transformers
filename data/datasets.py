from typing import List, Dict
from pathlib import Path
import pickle
import math


import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from data.tokenizer import MyTokenizer


class DecoderMLMDataset(Dataset):
    def __init__(
        self,
        tokenizer: MyTokenizer,
        text_folders: List[str],
        max_len: int = 512,
        limit: int = None,
        begin_sentence_token: int = 1,
        end_of_sentence_token: int = 0,
        pad_token_id: int = 2,
        vocab_size: int = 60000,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.begin_sentence_token = begin_sentence_token
        self.end_of_sentence_token = end_of_sentence_token
        self.pad_token_id = pad_token_id

        self.texts_refs = []
        for folder in text_folders:
            self.texts_refs += list(Path(folder).glob("**/*.txt"))

        if limit is not None:
            self.texts_refs = self.texts_refs[:limit]

    def __len__(self):
        return len(self.texts_refs)

    def __getitem__(self, index):
        text = self.texts_refs[index].read_text(encoding="utf8")
        tokenized = self.tokenizer.tokenize_text(text)
        # for this guys is important to put begin and end of sentence tokens
        tokenized = (
            [self.begin_sentence_token] + tokenized + [self.end_of_sentence_token]
        )
        # for the decoder we must have a maximum length of max_len + 1 for shifted values
        # its ok to pad but the padding must come from left ro rigth
        if len(tokenized) < self.max_len + 1:
            diff = (self.max_len + 1) - len(tokenized)
            tokenized = [self.pad_token_id for _ in range(diff)] + tokenized
        elif len(tokenized) > self.max_len + 1:
            tokenized = tokenized[: self.max_len + 1]

        # transform into tensor
        tokenized = torch.from_numpy(np.array(tokenized))

        decoder_input = tokenized[: self.max_len]
        desired_output = tokenized[1 : self.max_len + 1]

        return decoder_input, desired_output


class TokenizedDecoderMLMDataset(Dataset):
    def __init__(
        self,
        text_folders: List[str],
        max_len: int = 512,
        limit: int = None,
        begin_sentence_token: int = 1,
        end_of_sentence_token: int = 0,
        pad_token_id: int = 2,
        vocab_size: int = 60000,
    ):
        super().__init__()

        self.max_len = max_len
        self.vocab_size = vocab_size
        self.begin_sentence_token = begin_sentence_token
        self.end_of_sentence_token = end_of_sentence_token
        self.pad_token_id = pad_token_id

        self.texts_refs = []
        for folder in text_folders:
            self.texts_refs += list(Path(folder).glob("**/*.pkl"))

        if limit is not None:
            self.texts_refs = self.texts_refs[:limit]

    def __len__(self):
        return len(self.texts_refs)

    def __getitem__(self, index):
        tokenized = None
        with open(self.texts_refs[index], "rb") as file:
            tokenized = pickle.load(file)
        # for the decoder we must have a maximum length of max_len + 1 for shifted values
        # its ok to pad but the padding must come from left ro rigth
        if len(tokenized) < self.max_len + 1:
            diff = (self.max_len + 1) - len(tokenized)
            tokenized = [self.pad_token_id for _ in range(diff)] + tokenized
        elif len(tokenized) > self.max_len + 1:
            tokenized = tokenized[: self.max_len + 1]

        # transform into tensor
        tokenized = torch.from_numpy(np.array(tokenized))

        decoder_input = tokenized[: self.max_len]
        desired_output = tokenized[1 : self.max_len + 1]

        return decoder_input, desired_output


class EncoderMLMDataset(Dataset):
    def __init__(
        self,
        tokenizer: MyTokenizer,
        text_folders: List[str],
        max_len: int = 512,
        mask_prob: float = 0.15,
        limit: int = None,
        mask_token_id: int = 1,
        pad_token_id: int = 0,
        special_tokens: List[int] = [],
        vocab_size: int = 60000,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.special_tokens = special_tokens
        self.vocab_size = vocab_size

        self.texts_refs = []
        for folder in text_folders:
            self.texts_refs += list(Path(folder).glob("**/*.txt"))

        if limit is not None:
            self.texts_refs = self.texts_refs[:limit]

    def __len__(self):
        return len(self.texts_refs)

    def __getitem__(self, index):
        text = self.texts_refs[index].read_text(encoding="utf8")
        tokenized = self.tokenizer.tokenize_text(text)
        # before masking its nice to have everything in the right size
        if len(tokenized) < self.max_len:
            diff = self.max_len - len(tokenized)
            tokenized += [self.pad_token_id for _ in range(diff)]
        elif len(tokenized) > self.max_len:
            tokenized = tokenized[: self.max_len]
        # transform into tensor
        tokenized = torch.from_numpy(np.array(tokenized))
        # mask time
        probas = torch.rand(tokenized.shape)
        mask = (probas < self.mask_prob) * (tokenized != self.pad_token_id)
        # special tokens
        for special_token in self.special_tokens:
            mask = mask * (tokenized != special_token)

        # now mask tokenized with the mask we just created
        masked = torch.clone(tokenized).type(torch.int)
        masked_ids = torch.flatten(mask.nonzero())
        masked_ids_list = masked_ids.tolist()
        # 80% will be replaced by the mask token
        # 10% no change
        # 10% replaced by random token
        original_masked_tokens = tokenized[masked_ids_list]
        replace_masked_tokens = self.generate_mlm_tokens(
            original_masked_tokens.tolist()
        )
        masked[masked_ids_list] = replace_masked_tokens

        return masked, tokenized, mask

    def generate_mlm_tokens(self, original_tokens: List[int]):
        # 80% will be replaced by the mask token
        # 10% no change
        # 10% replaced by random token
        replace_tokens = np.random.rand(len(original_tokens))
        for i in range(len(original_tokens)):
            if replace_tokens[i] <= 0.8:
                replace_tokens[i] = self.mask_token_id
            elif replace_tokens[i] <= 0.9:
                replace_tokens[i] = np.random.randint(self.vocab_size)
            else:
                replace_tokens[i] = original_tokens[i]
        return torch.from_numpy(replace_tokens).type(torch.int)


class TokenizedEncoderMLMDataset(Dataset):
    def __init__(
        self,
        tokens_folders: List[str],
        max_len: int = 512,
        mask_prob: float = 0.15,
        limit: int = None,
        mask_token_id: int = 1,
        pad_token_id: int = 0,
        special_tokens: List[int] = [],
        vocab_size: int = 60000,
    ):
        super().__init__()

        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.special_tokens = special_tokens
        self.vocab_size = vocab_size

        self.tokens_ref = []
        for folder in tokens_folders:
            self.tokens_ref += list(Path(folder).glob("**/*.pkl"))

        if limit is not None:
            self.tokens_ref = self.tokens_ref[:limit]

    def __len__(self):
        return len(self.tokens_ref)

    def __getitem__(self, index):
        tokenized = None
        with open(self.tokens_ref[index], "rb") as tokens_file:
            tokenized = pickle.load(tokens_file)
        # before masking its nice to have everything in the right size
        if len(tokenized) < self.max_len:
            diff = self.max_len - len(tokenized)
            tokenized += [self.pad_token_id for _ in range(diff)]
        elif len(tokenized) > self.max_len:
            tokenized = tokenized[: self.max_len]
        # transform into tensor
        tokenized = torch.from_numpy(np.array(tokenized))
        # mask time
        probas = torch.rand(tokenized.shape)
        mask = (probas < self.mask_prob) * (tokenized != self.pad_token_id)
        # special tokens
        for special_token in self.special_tokens:
            mask = mask * (tokenized != special_token)

        # now mask tokenized with the mask we just created
        masked = torch.clone(tokenized).type(torch.int)
        masked_ids = torch.flatten(mask.nonzero())
        masked_ids_list = masked_ids.tolist()
        # 80% will be replaced by the mask token
        # 10% no change
        # 10% replaced by random token
        original_masked_tokens = tokenized[masked_ids_list]
        replace_masked_tokens = self.generate_mlm_tokens(
            original_masked_tokens.tolist()
        )
        masked[masked_ids_list] = replace_masked_tokens

        return masked, tokenized, mask

    def generate_mlm_tokens(self, original_tokens: List[int]):
        # 80% will be replaced by the mask token
        # 10% no change
        # 10% replaced by random token
        replace_tokens = np.random.rand(len(original_tokens))
        for i in range(len(original_tokens)):
            if replace_tokens[i] <= 0.8:
                replace_tokens[i] = self.mask_token_id
            elif replace_tokens[i] <= 0.9:
                replace_tokens[i] = np.random.randint(self.vocab_size)
            else:
                replace_tokens[i] = original_tokens[i]
        return torch.from_numpy(replace_tokens).type(torch.int)


class HBARTDataset(Dataset):
    def __init__(
        self,
        tokens_metadata: str,
        max_len: int = 32256,
        min_len: int = 512,
        batches: Dict = {512: 8, 1024: 4, 2048: 2, 4096: 1},
        limit: int = None,
        begin_sentence_token: int = 1,
        end_of_sentence_token: int = 0,
        pad_token_id: int = 2,
        tokens_decoder: int = 512,
        segment_size: int = 32,
    ):
        super().__init__()

        self.max_len = max_len
        self.min_len = min_len
        self.tokens_decoder = tokens_decoder
        self.segment_size = segment_size
        self.begin_sentence_token = begin_sentence_token
        self.end_of_sentence_token = end_of_sentence_token
        self.pad_token_id = pad_token_id
        self.batches_config = batches
        self.num_batches = None

        self.tokens_dataset = pd.read_parquet(tokens_metadata)
        # ja filtra por min len e max
        self.tokens_dataset = self.tokens_dataset[
            (self.tokens_dataset.token_bin <= max_len)
            & (self.tokens_dataset.token_bin >= min_len)
        ].reset_index(drop=True)

        if limit is not None:
            self.tokens_dataset = self.tokens_dataset.iloc[:limit]

        # seta os batches
        self.prepare_batches()

    def prepare_batches(self):
        # ordena o dataset
        self.tokens_dataset = self.tokens_dataset.sort_values(
            ["token_bin"], ascending=True
        ).reset_index(drop=True)

        batches = []
        batch_index = 0
        cur_batch_size = 0
        current_bin = self.tokens_dataset.token_bin.min()
        max_batch_size = self.get_max_batch_size(current_bin)
        for _, row in self.tokens_dataset.iterrows():
            token_bin = row.token_bin

            # ja finalizamos o batch?
            if cur_batch_size == max_batch_size or current_bin != token_bin:
                # vamos para um novo
                batch_index += 1
                cur_batch_size = 0
                current_bin = token_bin
                max_batch_size = self.get_max_batch_size(current_bin)

            # adiciona no batch
            batches.append(batch_index)
            cur_batch_size += 1

        self.tokens_dataset["batch_index"] = batches
        self.num_batches = batches[-1]

    def get_max_batch_size(self, token_len):
        max_batch_size = None
        for len, max_batch in self.batches_config.items():
            max_batch_size = max_batch
            if token_len <= len:
                break

        return max_batch_size

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch_info = self.tokens_dataset[self.tokens_dataset.batch_index == index]
        decoder_inputs = []
        labels = []
        encoder_inputs = []
        for _, row in batch_info.iterrows():
            token_ref = row.tokens_ref
            desired_len = row.token_bin
            token = None
            with open(token_ref, "rb") as file:
                token = pickle.load(file)

            # tem que separar o que vai para o encoder e o que vai para o decoder e o label
            decoder_input = token[-self.tokens_decoder - 1 : -1]
            label = token[-self.tokens_decoder :]
            encoder_input = token[: -self.tokens_decoder - 1]
            # decoder nao precisa de padding
            # label nao precisa de padding
            # for the decoder we must have a maximum length of max_len + 1 for shifted values
            # its ok to pad but the padding must come from left ro rigth
            # NESTE CASO ESPECIFICO VAMOS FAZER O PADDING DO ENCODER A ESQUERDA E NAO A DIREITA
            # para que os ultimos embeddings tenham significado e nao fiquem vazios
            # (no caso de ter que jogar novos embeddings gerados para o encoder em tempo de execucao)
            # tamanho tem que ser multiplo do segment size - 1
            desired_len_encoder = (
                math.ceil(len(decoder_input) / (self.segment_size - 1))
                * self.segment_size
            )
            if len(encoder_input) < desired_len_encoder + 1:
                diff = (desired_len_encoder + 1) - len(encoder_input)
                encoder_input = [self.pad_token_id for _ in range(diff)] + encoder_input
            elif len(encoder_input) > desired_len_encoder + 1:
                encoder_input = encoder_input[: desired_len_encoder + 1]

            # transform into tensor and add batch dimension
            decoder_input = torch.from_numpy(np.array(decoder_input)).unsqueeze(dim=0)
            label = torch.from_numpy(np.array(label)).unsqueeze(dim=0)
            encoder_input = torch.from_numpy(np.array(encoder_input)).unsqueeze(dim=0)

            decoder_inputs.append(decoder_input)
            labels.append(label)
            encoder_inputs.append(encoder_inputs)

        # vira tudo uma coisa so
        decoder_inputs = torch.stack(decoder_inputs, dim=0)
        labels = torch.stack(labels, dim=0)
        encoder_inputs = torch.stack(encoder_inputs, dim=0)

        return encoder_inputs, decoder_inputs, labels


class VariableLenEncoderMLMDataset(Dataset):
    def __init__(
        self,
        tokens_metadata: str,
        vocab_size: int,
        max_len: int = 32256,
        min_len: int = 512,
        batches: Dict = {512: 8, 1024: 4, 2048: 2, 4096: 1},
        limit: int = None,
        begin_sentence_token: int = 1,
        end_of_sentence_token: int = 0,
        pad_token_id: int = 2,
        segment_size: int = 32,
        mask_prob: float = 0.15,
        mask_token_id: int = 3,
        special_tokens: List[int] = [],
    ):
        super().__init__()

        self.max_len = max_len
        self.min_len = min_len
        self.segment_size = segment_size
        self.begin_sentence_token = begin_sentence_token
        self.end_of_sentence_token = end_of_sentence_token
        self.pad_token_id = pad_token_id
        self.batches_config = batches
        self.num_batches = None
        self.mask_prob = mask_prob
        self.special_tokens = special_tokens
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size

        self.tokens_dataset = pd.read_parquet(tokens_metadata)
        # ja filtra por min len e max
        # self.tokens_dataset = self.tokens_dataset[
        #     (self.tokens_dataset.num_tokens <= max_len)
        #     & (self.tokens_dataset.num_tokens >= min_len)
        # ].reset_index(drop=True)
        self.tokens_dataset = self.tokens_dataset[
            self.tokens_dataset.num_tokens >= min_len
        ].reset_index(drop=True)

        if limit is not None:
            self.tokens_dataset = self.tokens_dataset.iloc[:limit]

        # cria o token bin para este caso especifico
        token_bins = [
            (segment_size - 1) * i for i in range(max_len // segment_size + 1)
        ]
        my_token_bins = []
        for _, row in self.tokens_dataset.iterrows():
            num_tokens = row.num_tokens
            bin_index = 0
            while num_tokens > token_bins[bin_index]:
                bin_index += 1
                if bin_index >= len(token_bins):
                    bin_index = len(token_bins) - 1
                    break
            my_token_bins.append(token_bins[bin_index])
        self.tokens_dataset["token_bin"] = my_token_bins

        # seta os batches
        self.prepare_batches()

    def prepare_batches(self):
        # ordena o dataset
        self.tokens_dataset = self.tokens_dataset.sort_values(
            ["token_bin"], ascending=True
        ).reset_index(drop=True)

        batches = []
        batch_index = 0
        cur_batch_size = 0
        current_bin = self.tokens_dataset.token_bin.min()
        max_batch_size = self.get_max_batch_size(current_bin)
        for _, row in self.tokens_dataset.iterrows():
            token_bin = row.token_bin

            # ja finalizamos o batch?
            if cur_batch_size == max_batch_size or current_bin != token_bin:
                # vamos para um novo
                batch_index += 1
                cur_batch_size = 0
                current_bin = token_bin
                max_batch_size = self.get_max_batch_size(current_bin)

            # adiciona no batch
            batches.append(batch_index)
            cur_batch_size += 1

        self.tokens_dataset["batch_index"] = batches
        self.num_batches = batches[-1]

    def get_max_batch_size(self, token_len):
        max_batch_size = None
        for len, max_batch in self.batches_config.items():
            max_batch_size = max_batch
            if token_len <= len:
                break

        return max_batch_size

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch_info = self.tokens_dataset[
            self.tokens_dataset.batch_index == index
        ].reset_index(drop=True)
        masked_tensors = []
        tokenized_tensors = []
        mask_tensors = []
        for _, row in batch_info.iterrows():
            token_bin = row.token_bin
            token_ref = row.tokens_ref
            tokenized = None
            with open(token_ref, "rb") as file:
                tokenized = pickle.load(file)
            # decoder nao precisa de padding
            # label nao precisa de padding
            # for the decoder we must have a maximum length of max_len + 1 for shifted values
            # its ok to pad but the padding must come from left ro rigth
            # NESTE CASO ESPECIFICO VAMOS FAZER O PADDING DO ENCODER A ESQUERDA E NAO A DIREITA
            # para que os ultimos embeddings tenham significado e nao fiquem vazios
            # (no caso de ter que jogar novos embeddings gerados para o encoder em tempo de execucao)
            # tamanho tem que ser multiplo do segment size - 1
            desired_len_encoder = token_bin
            if len(tokenized) < desired_len_encoder:
                diff = (desired_len_encoder) - len(tokenized)
                tokenized = [self.pad_token_id for _ in range(diff)] + tokenized
            elif len(tokenized) > desired_len_encoder:
                tokenized = tokenized[:desired_len_encoder]

            # transform into tensor
            tokenized = torch.from_numpy(np.array(tokenized))
            # mask time
            probas = torch.rand(tokenized.shape)
            mask = (probas < self.mask_prob) * (tokenized != self.pad_token_id)
            # special tokens
            for special_token in self.special_tokens:
                mask = mask * (tokenized != special_token)

            # now mask tokenized with the mask we just created
            masked = torch.clone(tokenized).type(torch.int)
            masked_ids = torch.flatten(mask.nonzero())
            masked_ids_list = masked_ids.tolist()
            # 80% will be replaced by the mask token
            # 10% no change
            # 10% replaced by random token
            original_masked_tokens = tokenized[masked_ids_list]
            replace_masked_tokens = self.generate_mlm_tokens(
                original_masked_tokens.tolist()
            )
            masked[masked_ids_list] = replace_masked_tokens

            # adiciona a dimensao de batch em tudo e bota nas listas
            masked_tensors.append(masked)
            tokenized_tensors.append(tokenized)
            mask_tensors.append(mask)

        # vira tudo uma coisa so
        masked_tensors = torch.stack(masked_tensors, dim=0)
        tokenized_tensors = torch.stack(tokenized_tensors, dim=0)
        mask_tensors = torch.stack(mask_tensors, dim=0)

        return masked_tensors, tokenized_tensors, mask_tensors

    def generate_mlm_tokens(self, original_tokens: List[int]):
        # 80% will be replaced by the mask token
        # 10% no change
        # 10% replaced by random token
        replace_tokens = np.random.rand(len(original_tokens))
        for i in range(len(original_tokens)):
            if replace_tokens[i] <= 0.8:
                replace_tokens[i] = self.mask_token_id
            elif replace_tokens[i] <= 0.9:
                replace_tokens[i] = np.random.randint(self.vocab_size)
            else:
                replace_tokens[i] = original_tokens[i]
        return torch.from_numpy(replace_tokens).type(torch.int)


class VariableLenEncoderDecoderReconstruction(Dataset):
    def __init__(
        self,
        tokens_metadata: str,
        vocab_size: int,
        max_len: int = 32256,
        min_len: int = 512,
        batches: Dict = {512: 8, 1024: 4, 2048: 2, 4096: 1},
        limit: int = None,
        begin_sentence_token: int = 1,
        end_of_sentence_token: int = 0,
        pad_token_id: int = 2,
        segment_size: int = 32,
        mask_prob: float = 0.15,
        mask_token_id: int = 3,
        special_tokens: List[int] = [],
    ):
        super().__init__()

        self.max_len = max_len
        self.min_len = min_len
        self.segment_size = segment_size
        self.begin_sentence_token = begin_sentence_token
        self.end_of_sentence_token = end_of_sentence_token
        self.pad_token_id = pad_token_id
        self.batches_config = batches
        self.num_batches = None
        self.mask_prob = mask_prob
        self.special_tokens = special_tokens
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size

        self.tokens_dataset = pd.read_parquet(tokens_metadata)
        # ja filtra por min len e max
        # self.tokens_dataset = self.tokens_dataset[
        #     (self.tokens_dataset.num_tokens <= max_len)
        #     & (self.tokens_dataset.num_tokens >= min_len)
        # ].reset_index(drop=True)
        self.tokens_dataset = self.tokens_dataset[
            self.tokens_dataset.num_tokens >= min_len
        ].reset_index(drop=True)

        if limit is not None:
            self.tokens_dataset = self.tokens_dataset.iloc[:limit]

        # cria o token bin para este caso especifico
        token_bins = [
            (segment_size - 1) * i for i in range(max_len // segment_size + 1)
        ]
        my_token_bins = []
        for _, row in self.tokens_dataset.iterrows():
            num_tokens = row.num_tokens
            bin_index = 0
            while num_tokens > token_bins[bin_index]:
                bin_index += 1
                if bin_index >= len(token_bins):
                    bin_index = len(token_bins) - 1
                    break
            my_token_bins.append(token_bins[bin_index])
        self.tokens_dataset["token_bin"] = my_token_bins

        # seta os batches
        self.prepare_batches()

    def prepare_batches(self):
        # ordena o dataset
        self.tokens_dataset = self.tokens_dataset.sort_values(
            ["token_bin"], ascending=True
        ).reset_index(drop=True)

        batches = []
        batch_index = 0
        cur_batch_size = 0
        current_bin = self.tokens_dataset.token_bin.min()
        max_batch_size = self.get_max_batch_size(current_bin)
        for _, row in self.tokens_dataset.iterrows():
            token_bin = row.token_bin

            # ja finalizamos o batch?
            if cur_batch_size == max_batch_size or current_bin != token_bin:
                # vamos para um novo
                batch_index += 1
                cur_batch_size = 0
                current_bin = token_bin
                max_batch_size = self.get_max_batch_size(current_bin)

            # adiciona no batch
            batches.append(batch_index)
            cur_batch_size += 1

        self.tokens_dataset["batch_index"] = batches
        self.num_batches = batches[-1]

    def get_max_batch_size(self, token_len):
        max_batch_size = None
        for len, max_batch in self.batches_config.items():
            max_batch_size = max_batch
            if token_len <= len:
                break

        return max_batch_size

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch_info = self.tokens_dataset[
            self.tokens_dataset.batch_index == index
        ].reset_index(drop=True)
        tokenized_tensors_encoder = []
        tokenized_tensors_decoder = []
        tokenized_tensors_labels = []
        for _, row in batch_info.iterrows():
            token_bin = row.token_bin
            token_ref = row.tokens_ref
            tokenized = None
            with open(token_ref, "rb") as file:
                tokenized = pickle.load(file)
            # decoder nao precisa de padding
            # label nao precisa de padding
            # for the decoder we must have a maximum length of max_len + 1 for shifted values
            # its ok to pad but the padding must come from left ro rigth
            # NESTE CASO ESPECIFICO VAMOS FAZER O PADDING DO ENCODER A ESQUERDA E NAO A DIREITA
            # para que os ultimos embeddings tenham significado e nao fiquem vazios
            # (no caso de ter que jogar novos embeddings gerados para o encoder em tempo de execucao)
            # tamanho tem que ser multiplo do segment size - 1
            desired_len_encoder = token_bin
            if len(tokenized) < desired_len_encoder:
                diff = (desired_len_encoder) - len(tokenized)
                tokenized = [self.pad_token_id for _ in range(diff)] + tokenized
            elif len(tokenized) > desired_len_encoder:
                # pega um pedaco aleatorio dentre os possiveis
                possible_starts = len(tokenized) - desired_len_encoder
                random_start = np.random.randint(0, possible_starts)
                tokenized = tokenized[random_start : random_start + desired_len_encoder]

            # transform into tensor
            tokenized = torch.from_numpy(np.array(tokenized))
            tokenized_tensors_encoder.append(tokenized)

            # agora as coisas relacionadas ao decoder
            tokenized_tensors_decoder.append(tokenized[0:-1])
            tokenized_tensors_labels.append(tokenized[1:])

        # vira tudo uma coisa so
        tokenized_tensors_encoder = torch.stack(tokenized_tensors_encoder, dim=0)
        tokenized_tensors_decoder = torch.stack(tokenized_tensors_decoder, dim=0)
        tokenized_tensors_labels = torch.stack(tokenized_tensors_labels, dim=0)

        return (
            tokenized_tensors_encoder,
            tokenized_tensors_decoder,
            tokenized_tensors_labels,
        )
