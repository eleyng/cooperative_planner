""" Cooperative VRNN Planner training | evaluation """
import sys
from os import mkdir
from os.path import isdir, isfile, join

sys.path.append("./")

import pytorch_lightning as pl

from configs.exp_config import get_args
from models.bcrnn import BCRNN
from models.vrnn import VRNN
from utils.loaders import RolloutDataModule


def main(sysargv):

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
    elif config.model == "bcrnn":
        model = BCRNN(config)
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
        wb_logger = pl.loggers.WandbLogger(
            project=config.project,
            name=config.name,
            log_model="all",
        )
    else:
        wb_logger = pl.loggers.WandbLogger(
            project=config.project,
            log_model="all",
        )

    # ------------------------
    # 4 TRAINER
    # ------------------------
    # Create trainer

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
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
        deterministic="warn",
    )

    # ------------------------
    # 5 START TRAINING
    # ------------------------
    # Train model. You can optionally restore a model from a checkpoint by setting the restore flag to True,
    # and indicating the path to the checkpoint in the artifact_path argument.

    if config.restore:

        assert isfile(join("trained_models", config.artifact_path)), "Artifact not found"

        if config.model == "bcrnn":
            model = BCRNN.load_from_checkpoint(
                join("trained_models", config.artifact_path),
                hparams=config,
            )
        elif config.model == "vrnn":
            model = VRNN.load_from_checkpoint(
                join("trained_models", config.artifact_path),
                hparams=config,
            )
        else:
            raise ValueError("Model not supported")
        print("Restoring model from: ", config.artifact_path)

        if not config.train:
            model.eval()
            print("Test model from: ", config.artifact_path)
            trainer.test(
                model,
                ckpt_path=join("trained_models", config.artifact_path),
                dataloaders=test_data_loader,
            )

        else:
            # Load restored model to resume training
            print("Resuming training model from: ", config.artifact_path)

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
    main(sys.argv)
