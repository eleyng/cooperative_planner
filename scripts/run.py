""" Cooperative VRNN Planner training | evaluation """
from os import mkdir
from os.path import join, isdir
import sys

sys.path.append("./")

import pytorch_lightning as pl
from utils.loaders import RolloutDataModule
from models.vrnn import VRNN
from configs.exp_config import get_args


def main():

    config = get_args()

    # ------------------------
    # 0 CONFIG PATHS
    # ------------------------
    # Create results & plots dir
    results_dir = config.results_dir
    plot_dir = config.plot_dir
    if not isdir(results_dir):
        mkdir(results_dir)
    plot_base_dir = join(results_dir, plot_dir)
    if not isdir(plot_base_dir):
        mkdir(plot_base_dir)
    # Find data dir
    dataset_dir = join("datasets", config.data_dir)
    print("dataset: ", dataset_dir)
    assert isdir(dataset_dir), "Dataset not found"

    # ------------------------
    # 1 LIGHTNING MODEL
    # ------------------------
    # Create model
    pl.seed_everything(config.seed, workers=True)

    if config.model == "vrnn":
        model = VRNN(config)
    else:
        raise ValueError("Model not supported")

    # ------------------------
    # 2 DATA PIPELINES
    # ------------------------
    # Create datamodule

    datamodule = RolloutDataModule(
        config.transform,
        dataset_dir,
        seq_len=config.SEQ_LEN,
        H=config.H,
        skip=config.skip,
        batch_size=config.BSIZE,
        num_workers=config.num_workers,
        test_data=config.test_data,
    )

    train_data_loader = datamodule.train_dataloader()
    val_data_loader = datamodule.val_dataloader()
    test_data_loader = datamodule.test_dataloader()

    # ------------------------
    # 3 WANDB LOGGER
    # ------------------------

    if config.restore:
        wb_logger = pl.WandbLogger(
            project=config.project,
            name=config.experiment_name,
            log_model="all",
            id=config.id,
        )
    else:
        wb_logger = pl.WandbLogger(
            project=config.project,
            log_model="all",
        )

    # ------------------------
    # 4 TRAINER
    # ------------------------
    # Create trainer

    lr_monitor = pl.LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min")

    trainer = pl.Trainer(
        default_root_dir="trained_models",
        accelerator=config.device,
        gradient_clip_val=config.grad_clip_val,
        track_grad_norm=2,
        strategy="ddp",
        max_epochs=config.epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=[wb_logger],
        deterministic=True,
    )

    # ------------------------
    # 5 START TRAINING
    # ------------------------
    # Train model. You can optionally restore a model from a checkpoint by setting the restore flag to True,
    # and indicating the path to the checkpoint in the artifact_dir argument.

    if config.restore:

        assert isdir(join("trained_models", config.artifact_dir)), "Artifact not found"
        model = VRNN.load_from_checkpoint(
            join("trained_models", config.artifact_dir),
            hparams=config,
        )
        print("Restoring model from: ", config.artifact_dir)

        if not config.train:
            model.eval()
            print("Test model from: ", config.artifact_dir)
            trainer.test(
                model,
                ckpt_path=config.artifact_dir,
                dataloaders=test_data_loader,
            )

        else:
            # Load restored model to resume training
            print("Resuming training model from: ", config.artifact_dir)

            trainer = pl.Trainer(
                default_root_dir="trained_models",
                accelerator=config.device,
                gradient_clip_val=config.grad_clip_val,
                track_grad_norm=2,
                strategy="ddp",
                max_epochs=config.epochs,
                logger=[wb_logger],
                callbacks=[checkpoint_callback, lr_monitor],
                resume_from_checkpoint=config.artifact_dir,
                deterministic=True,
            )
            trainer.fit(
                model,
                train_dataloaders=train_data_loader,
                val_dataloaders=val_data_loader,
            )

    else:
        # Train from scratch
        print("Training model from scratch.")
        trainer.fit(
            model, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader
        )


if __name__ == "__main__":
    main()
