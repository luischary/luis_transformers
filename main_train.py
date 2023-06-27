import argparse
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import torch

from data.datasets import (
    TokenizedDecoderMLMDataset,
    TokenizedEncoderMLMDataset,
    VariableLenEncoderMLMDataset,
    VariableLenEncoderDecoderReconstruction,
)
from data.tokenizer import MyTokenizer

from transformer.vanilla.encoder_lightining import VanillaEncoderMLMLightning
from transformer.hierarchical.hierarchical_lightining import (
    HierarchicalEncoderMLMLightning,
)
from transformer.decoder.myGPTL import MyGPTLightning

# from transformer.decoder.myGPTL import MyGPTLightning
from transformer.HBART.lightning_model import HBARTLMLightning
from configs import vanilla_encoder_params, decoder_params, hierarchical_encoder_params

torch.backends.cuda.matmul.allow_tf32 = True


def get_dataset(args):
    max_len_dataset = args.tokens_max_len
    min_len_dataset = args.tokens_min_len
    if args.model_type == "bart" or (
        args.model_type == "encoder" and args.encoder_type == "hierarchical"
    ):
        segment_size = args.segment_size
        max_len_dataset = max_len_dataset - (max_len_dataset // segment_size)
        min_len_dataset = min_len_dataset - (min_len_dataset // segment_size)

    if args.model_type == "encoder":
        if args.dataset_type == "regular":
            dataset = TokenizedEncoderMLMDataset(
                tokens_folders=["data/tokenized_datasets/tokens_marketplace_128"],
                max_len=max_len_dataset,
                mask_token_id=3,
                pad_token_id=0,
                special_tokens=[0, 1, 2, 3, 4, 5, 6, 7],
                vocab_size=60000,
            )
        else:
            dataset = VariableLenEncoderMLMDataset(
                tokens_metadata="data/infos_tokens.pq",
                max_len=max_len_dataset,
                min_len=min_len_dataset,
                batches={500: 16, 1000: 8, 2000: 4, 4000: 2, 8000: 1},
                limit=args.dataset_limit,
                begin_sentence_token=1,
                end_of_sentence_token=2,
                pad_token_id=0,
                segment_size=args.segment_size,
                mask_prob=0.15,
                special_tokens=[0, 1, 2, 3, 4, 5, 6, 7],
                mask_token_id=3,
                vocab_size=50000,
            )
    elif args.model_type == "decoder":
        dataset = TokenizedDecoderMLMDataset(
            text_folders=["data/tokenized_datasets/tokens_marketplace_128"],
            max_len=max_len_dataset,
            limit=args.dataset_limit,
            begin_sentence_token=1,
            end_of_sentence_token=2,
            pad_token_id=0,
            vocab_size=60000,
        )
    elif args.model_type == "bart":
        dataset = VariableLenEncoderDecoderReconstruction(
            tokens_metadata="data/infos_tokens.pq",
            max_len=max_len_dataset,
            min_len=min_len_dataset,
            batches={
                256 - (256 // 32): 8,
                512 - (512 // 32): 4,
                1024 - (1024 // 32): 2,
                2048 - (2048 // 32): 1,
            },
            limit=args.dataset_limit,
            begin_sentence_token=1,
            end_of_sentence_token=2,
            pad_token_id=0,
            segment_size=args.segment_size,
            mask_prob=0.15,
            special_tokens=[0, 1, 2, 3, 4, 5, 6, 7],
            mask_token_id=3,
            vocab_size=50000,
        )
    return dataset


def get_modelo(args):
    if args.model_type == "encoder":
        if args.encoder_type == "vanilla":
            modelo = VanillaEncoderMLMLightning(
                encoder_params=vanilla_encoder_params,
                learning_rate=args.lr,
                min_lr_percent=args.min_lr_percent,
                warmup_steps=args.warmup_lr_steps,
                total_training_steps=args.training_steps,
            )
        else:
            modelo = HierarchicalEncoderMLMLightning(
                encoder_params=hierarchical_encoder_params,
                learning_rate=args.lr,
                min_lr_percent=args.min_lr_percent,
                warmup_steps=args.warmup_lr_steps,
                total_training_steps=args.training_steps,
            )
    elif args.model_type == "decoder":
        modelo = MyGPTLightning(
            decoder_params=decoder_params,
            learning_rate=args.lr,
            min_lr_percent=args.min_lr_percent,
            warmup_steps=args.warmup_lr_steps,
            total_training_steps=args.training_steps,
        )
    elif args.model_type == "bart":
        modelo = HBARTLMLightning(
            vocab_size=50000,
            embed_dim=1280,
            encoder_params=hierarchical_encoder_params,
            decoder_params=decoder_params,
            learning_rate=args.lr,
            min_lr_percent=args.min_lr_percent,
            warmup_steps=args.warmup_lr_steps,
            total_training_steps=args.training_steps,
            reconstruction=True,
        )

    return modelo


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument(
        "--model-name",
        help="Nome do modelo que sera treinado",
        type=str,
        default="teste",
    )
    parser.add_argument(
        "--model-path",
        help="Caminho onde deseja salvar os binarios relacionados ao modelo",
        type=str,
        default="modelos_treinados",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="encoder",
        help="Tipo do modelo: encoder, decoder ou bart",
        choices=["encoder", "decoder", "bart"],
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="regular",
        help="Tipo de dataset. Regular ou variable (length)",
        choices=["regular", "variable"],
    )
    parser.add_argument(
        "--encoder-type",
        type=str,
        default="vanilla",
        help="Tipo de encoder. Vanilla ou hierarchical",
        choices=["vanilla", "hierarchical"],
    )
    parser.add_argument(
        "--tokens-max-len",
        type=int,
        default=128,
        help="Maior sequencia de tokens que o dataset entregara para o modelo",
    )
    parser.add_argument(
        "--tokens-min-len",
        type=int,
        default=32,
        help="Menor sequencia de tokens que o dataset entregara para o modelo",
    )
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=None,
        help="Limite do tamanho do dataset (filtro de metadados)",
    )
    parser.add_argument(
        "--segment-size",
        type=int,
        default=32,
        help="Tamahho do segmento do encoder hierarquico",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate base para treinamento",
    )
    parser.add_argument(
        "--min-lr-percent",
        type=float,
        default=0.1,
        help="Valor minimo final do learning rate atraves do scheduler",
    )
    parser.add_argument(
        "--warmup-lr-steps",
        type=int,
        default=1000,
        help="Quantidade de steps para o warmup do learning rate",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=100_000,
        help="Maxima quantidade de steps de treinamento",
    )
    parser.add_argument(
        "--accum-batch",
        type=int,
        default=1,
        help="Quantidade de batches para acumular",
    )
    parser.add_argument(
        "--num-data-workers",
        type=int,
        default=1,
        help="Quantidade de workers do dataloader",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Tamanho do batch carregado pelo dataloader",
    )

    # Read arguments from command line
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    dataset = get_dataset(args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_data_workers,
        shuffle=True,
        batch_size=args.batch_size,
    )

    modelo = get_modelo(args)

    # modelo = modelo.load_from_checkpoint(
    #     "modelos_treinados/BART/general/hbart_reconstruction/last.ckpt"
    # )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.model_path + "/" + args.model_name,
        filename=args.model_name + "_{epoch}-{step}",
        save_last=True,
        every_n_train_steps=1000,
        monitor="train_loss",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        accelerator=device,
        max_steps=args.training_steps,
        log_every_n_steps=25,
        num_sanity_val_steps=3,
        accumulate_grad_batches=args.accum_batch,
        precision="16-mixed",
        enable_progress_bar=True,
        callbacks=[checkpoint_callback],
        default_root_dir=args.model_path + "/" + args.model_name,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
        # strategy="deepspeed_stage_2_offload",
        # detect_anomaly=True,
        # strategy=DeepSpeedStrategy(
        #     offload_optimizer=True, allgather_bucket_size=5e8, reduce_bucket_size=5e8
        # ),
    )
    trainer.fit(model=modelo, train_dataloaders=dataloader)
