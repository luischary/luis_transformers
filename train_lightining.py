import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DeepSpeedStrategy
import torch

from data.datasets import (
    TokenizedDecoderMLMDataset,
    TokenizedEncoderMLMDataset,
    VariableLenEncoderMLMDataset,
)
from data.tokenizer import MyTokenizer

# from transformer.alibi.alibi_lightining import AlibiDecoderLightning
# from transformer.longformer.longformer_lightining import LongformerMLMLightning
# from transformer.vanilla.encoder_lightining import VanillaEncoderMLMLightning
from transformer.hierarchical.hierarchical_lightining import (
    HierarchicalEncoderMLMLightning,
)

# from transformer.decoder.myGPTL import MyGPTLightning
from transformer.HBART.lightning_model import HBARTLMLightning

torch.backends.cuda.matmul.allow_tf32 = True

if __name__ == "__main__":
    vocab_size = 50000
    max_len = 128
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tokenizer = MyTokenizer(tokenizer_path="artifacts/general_tokenizer_50")
    # tokenizer = MyTokenizer(tokenizer_path="artifacts/movies_tokenizer")
    tokenizer = MyTokenizer(tokenizer_path="artifacts/marketplace_tokenizer")

    # dataset = TokenizedDecoderMLMDataset(
    #     text_folders=["data/tokenized_datasets/tokens_marketplace_128"],
    #     max_len=max_len,
    #     limit=None,
    #     begin_sentence_token=1,
    #     end_of_sentence_token=2,
    #     pad_token_id=0,
    #     vocab_size=60000,
    # )

    # dataset = TokenizedEncoderMLMDataset(
    #     tokens_folders=["data/tokenized_datasets/tokens_marketplace_128"],
    #     max_len=124,
    #     mask_token_id=3,
    #     pad_token_id=0,
    #     special_tokens=[0, 1, 2, 3, 4, 5, 6, 7],
    #     vocab_size=60000,
    # )

    dataset = VariableLenEncoderMLMDataset(
        tokens_metadata="data/infos_tokens.pq",
        max_len=8192 - (8192 // 32),
        min_len=496,
        batches={500: 16, 1000: 8, 2000: 4, 4000: 2, 8000: 1},
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

    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=0, shuffle=True, batch_size=1
    )

    # decoder_params = {
    #     "vocab_size": 60000,
    #     "embed_dim": 256,
    #     "num_layers": 12,
    #     "num_heads": 8,
    #     "hidden_size": 256 * 4,
    #     "dropout": 0.1,
    #     "pad_token_id": 0,
    #     "tokens_per_sample": 128,
    #     "encoding_type": "exponential",
    # }
    # modelo = AlibiDecoderLightning(
    #     decoder_params=decoder_params,
    #     learning_rate=1e-4,
    #     min_lr_percent=0.1,
    #     warmup_steps=1000,
    #     total_training_steps=1500 * 20,
    # )

    # encoder_params = {
    #     "vocab_size": 50000,
    #     "embed_dim": 768,
    #     "hidden_size": 768 * 4,
    #     "num_heads": 12,
    #     "num_layers": 6,
    #     # "window_size": 32,
    #     "max_context_size": 1024,
    #     "dropout": 0.1,
    # }

    # modelo = LongformerMLMLightning(
    #     encoder_params=encoder_params,
    #     learning_rate=1e-3,
    #     min_lr_percent=0.01,
    #     warmup_steps=100,
    #     total_training_steps=525 * 20,
    # )

    # modelo = VanillaEncoderMLMLightning(
    #     encoder_params=encoder_params,
    #     learning_rate=1e-4,
    #     min_lr_percent=0.01,
    #     warmup_steps=10000,
    #     total_training_steps=200000,
    # )

    encoder_params = {
        "vocab_size": 50000,
        "embed_dim": 1280,
        "hidden_size": 1280 * 4,
        "num_heads": 20,
        # "max_context_size": 1024,
        "dropout": 0.1,
        "segment_size": 32,
        "global_token_id": 4,
        "layers_type": [
            "segment",
            "global",
            "segment",
            "global",
            "segment",
            "global",
            "segment",
            "global",
            "segment",
            "global",
            "segment",
            "global",
            "segment",
            "global",
            "segment",
            "global",
            "segment",
            "global",
            "segment",
            "global",
            "segment",
            "global",
            "segment",
            "global",
        ],
    }
    modelo = HierarchicalEncoderMLMLightning(
        encoder_params=encoder_params,
        learning_rate=1e-4,
        min_lr_percent=0.01,
        warmup_steps=10000,
        total_training_steps=500000,
    )
    # decoder_params = {
    #     "vocab_size": 60000,
    #     "embed_dim": 256,
    #     "num_layers": 6,
    #     "num_heads": 8,
    #     "hidden_size": 256 * 4,
    #     "dropout": 0.1,
    # }
    # modelo = MyGPTLightning(
    #     decoder_params=decoder_params,
    #     learning_rate=1e-3,
    #     min_lr_percent=0.01,
    #     warmup_steps=1000,
    #     total_training_steps=40000,
    # )
    # modelo = HBARTLMLightning(
    #     vocab_size=60000,
    #     embed_dim=256,
    #     encoder_params=encoder_params,
    #     decoder_params=decoder_params,
    #     learning_rate=1e-3,
    #     min_lr_percent=0.01,
    #     warmup_steps=1000,
    #     total_training_steps=50000,
    # )
    model_name = "teste_hencoder_deepspeed"

    checkpoint_callback = ModelCheckpoint(
        dirpath="modelos_treinados/encoders/general/" + model_name,
        filename=model_name + "_{epoch}-{step}",
        save_last=True,
        every_n_train_steps=1000,
        monitor="train_loss",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        accelerator=device,
        max_steps=500000,
        log_every_n_steps=25,
        num_sanity_val_steps=3,
        accumulate_grad_batches=2,
        precision="16-mixed",
        enable_progress_bar=True,
        callbacks=[checkpoint_callback],
        default_root_dir="modelos_treinados/encoders/general/" + model_name,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
        strategy="deepspeed_stage_2",
        # detect_anomaly=True,
        # strategy=DeepSpeedStrategy(
        #     offload_optimizer=True, allgather_bucket_size=5e8, reduce_bucket_size=5e8
        # ),
    )
    trainer.fit(model=modelo, train_dataloaders=dataloader)
