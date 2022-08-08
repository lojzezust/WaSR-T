import argparse
import os
import math
import json
import torch
from torch.utils.data import DataLoader, ConcatDataset, BatchSampler, DistributedSampler
import torchvision.transforms as T
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from wasr_t.wasr_t import wasr_temporal_resnet101
from wasr_t.train import LitModel
from wasr_t.utils import MainLoggerCollection, Option
from wasr_t.callbacks import ModelExport
from wasr_t.data.mastr import MaSTr1325Dataset
from wasr_t.data.transforms import get_augmentation_transform, PytorchHubNormalization
from wasr_t.data.sampling import DatasetRandomSampler, DistributedSamplerWrapper, DatasetBatchSampler


# Enable/disable WANDB logging
WANDB_LOGGING = False

DEVICE_BATCH_SIZE = 3
TRAIN_CONFIG = 'configs/mastr1325_train.yaml'
VAL_CONFIG = 'configs/mastr1325_val.yaml'
NUM_CLASSES = 3
PATIENCE = 5
LOG_STEPS = 20
NUM_WORKERS = 1
NUM_GPUS = -1 # All visible GPUs
NUM_NODES = 1 # Single node training
RANDOM_SEED = None
OUTPUT_DIR = 'output'
PRETRAINED_DEEPLAB = True
PRECISION = 32
MONITOR_VAR = 'val/iou/obstacle'
MONITOR_VAR_MODE = 'max'
ADDITONAL_SAMPLES_RATIO = 0.5

HIST_LEN = 5
BACKBONE_GRAD_STEPS = 2

def get_arguments(input_args=None):
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=DEVICE_BATCH_SIZE,
                        help="Minibatch size (number of samples) used on each device.")
    parser.add_argument("--train-config", type=str, default=TRAIN_CONFIG,
                        help="Path to the file containing the training dataset config.")
    parser.add_argument("--additional-train-config", type=str, default=None,
                    help="Additional training config file. Can be used to sample from MaSTr1325 and MaSTr153 (additional) with equal probability.")
    parser.add_argument("--additional-samples-ratio", type=float, default=ADDITONAL_SAMPLES_RATIO,
                    help="(if < 1): Percentage of the batch to fill with additional training samples.\n"
                         "(if >= 1): Number of additional training samples in a batch.")
    parser.add_argument("--val-config", type=str, default=VAL_CONFIG,
                        help="Path to the file containing the val dataset config.")
    parser.add_argument("--mask-dir", type=str, default=None,
                        help="Override the original mask dir. Relative path from the dataset root.")
    parser.add_argument("--validation", action="store_true",
                        help="Report performance on validation set and use early stopping.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--patience", type=Option(int), default=PATIENCE,
                        help="Patience for early stopping (how many epochs to wait without increase).")
    parser.add_argument("--log-steps", type=int, default=LOG_STEPS,
                        help="Number of steps between logging variables.")
    parser.add_argument("--visualization-steps", type=int, default=None,
                        help="Number of steps between visualizing predictions. Default: only visualize at the end of training.")
    parser.add_argument("--num_nodes", type=int, default=NUM_NODES,
                        help="Number of nodes used for training.")
    parser.add_argument("--gpus", default=NUM_GPUS,
                        help="Number of gpus (or GPU ids) used for training.")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS,
                        help="Number of workers used for data loading.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--pretrained", type=bool, default=PRETRAINED_DEEPLAB,
                        help="Use pretrained DeepLab weights.")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Directory where the output will be stored (models and logs)")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Name of the model. Used to create model and log directories inside the output directory.")
    parser.add_argument("--pretrained-weights", type=str, default=None,
                        help="Path to the pretrained weights to be used.")
    parser.add_argument("--monitor-metric", type=str, default=MONITOR_VAR,
                        help="Validation metric to monitor for early stopping and best model saving.")
    parser.add_argument("--monitor-metric-mode", type=str, default=MONITOR_VAR_MODE, choices=['min', 'max'],
                        help="Maximize or minimize the monitored metric.")
    parser.add_argument("--no-augmentation", action="store_true",
                        help="Disable on-the-fly image augmentation of the dataset.")
    parser.add_argument("--precision", default=PRECISION, type=int, choices=[16,32],
                        help="Floating point precision.")
    parser.add_argument("--hist-len", default=HIST_LEN, type=int,
                        help="Number of past frames to be considered in addition to the target frame (context length).")
    parser.add_argument("--backbone-grad-steps", default=BACKBONE_GRAD_STEPS, type=int,
                        help="How far into the past the backbone gradients are propagated. 1 means gradients are only propagated through the target frame.")

    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume training from specified checkpoint.")

    parser = LitModel.add_argparse_args(parser)

    args = parser.parse_args(input_args)
    args = LitModel.parse_args(args)

    return args

class DataModule(pl.LightningDataModule):
    def __init__(self, args, normalize_t):
        super().__init__()
        self.args = args
        self.normalize_t = normalize_t

    def train_dataloader(self):
        transform = None
        if not self.args.no_augmentation:
            transform = get_augmentation_transform()

        # If using mask filling, use alternative mask subdir
        alternative_mask_subdir = None
        if self.args.mask_dir is not None:
            alternative_mask_subdir = self.args.mask_dir

        train_ds = MaSTr1325Dataset(self.args.train_config, transform=transform,
                                    normalize_t=self.normalize_t, masks_subdir=alternative_mask_subdir)

        b_sampler = None
        # Additional training file (combining multiple datasets)
        if self.args.additional_train_config is not None:
            orig_ds = train_ds
            add_ds = MaSTr1325Dataset(self.args.additional_train_config, transform=transform,
                                    normalize_t=self.normalize_t, masks_subdir=alternative_mask_subdir)
            train_ds = ConcatDataset([train_ds, add_ds])
            sample_ratio = self.args.additional_samples_ratio

            bs = self.args.batch_size
            if sample_ratio < 1:
                # Random sampling with probability
                orig_ratio = 1 - sample_ratio
                n_samples = int((1/orig_ratio) * len(orig_ds))
                sampler = DatasetRandomSampler(train_ds, [orig_ratio, sample_ratio], n_samples, shuffle=True)
                b_sampler = BatchSampler(sampler, bs, drop_last=True)
                b_sampler = DistributedSamplerWrapper(b_sampler)
            else:
                # Fixed number of samples per batch
                n_samples = int(sample_ratio)
                # Set number of batches so that all samples from the main dataset are observed
                n_batches = math.ceil(len(orig_ds) / (bs-n_samples))
                sampler = DatasetBatchSampler(train_ds, [bs-n_samples, n_samples], shuffle=True, num_batches=n_batches)
                b_sampler = DistributedSamplerWrapper(sampler)

        if b_sampler is None:
            sampler = DistributedSampler(train_ds, shuffle=True)
            train_dl = DataLoader(train_ds, batch_size=self.args.batch_size, sampler=sampler,
                                  num_workers=self.args.workers, drop_last=True)
        else:
            train_dl = DataLoader(train_ds, batch_sampler=b_sampler, num_workers=self.args.workers)

        return train_dl

    def val_dataloader(self):
        val_dl = None
        if self.args.validation:
            val_ds = MaSTr1325Dataset(self.args.val_config, normalize_t=self.normalize_t, include_original=True)
            val_dl = DataLoader(val_ds, batch_size=self.args.batch_size, num_workers=self.args.workers)

        return val_dl


def train_wasrt(args):
    # Use or create random seed
    args.random_seed = pl.seed_everything(args.random_seed)

    normalize_t = PytorchHubNormalization()
    data = DataModule(args, normalize_t)

    # Get model
    model = wasr_temporal_resnet101(num_classes=args.num_classes, pretrained=args.pretrained, hist_len=args.hist_len, backbone_grad_steps=args.backbone_grad_steps)

    if args.pretrained_weights is not None:
        print(f"Loading weights from: {args.pretrained_weights}")
        state_dict = torch.load(args.pretrained_weights, map_location='cpu')
        if 'model' in state_dict:
            # Loading weights from checkpoint
            model.load_state_dict(state_dict['model'])
        else:
            model.load_state_dict(state_dict)

    model = LitModel(model, args.num_classes, args)

    logs_path = os.path.join(args.output_dir, 'logs')
    tb_logger = pl_loggers.TensorBoardLogger(logs_path, args.model_name)
    loggers = [tb_logger]

    if WANDB_LOGGING:
        version_name = '%s/version_%d' % (args.model_name, tb_logger.version)
        wandb_logger = pl_loggers.WandbLogger(version_name, args.output_dir, project='WaSR', log_model=False)
        loggers.append(wandb_logger)

    logger = MainLoggerCollection(loggers)
    logger.log_hyperparams(args)

    callbacks = []
    if args.validation:
        # Val: Early stopping and best model saving
        if args.patience is not None:
            callbacks.append(EarlyStopping(monitor=args.monitor_metric, patience=args.patience, mode=args.monitor_metric_mode))
        callbacks.append(ModelCheckpoint(save_last=True, save_top_k=1, monitor=args.monitor_metric, mode=args.monitor_metric_mode))

    callbacks.append(ModelExport(os.path.join(args.output_dir, 'models')))

    trainer = pl.Trainer(logger=logger,
                         gpus=args.gpus,
                         num_nodes=args.num_nodes,
                         max_epochs=args.epochs,
                         accelerator='ddp',
                         resume_from_checkpoint=args.resume_from,
                         callbacks=callbacks,
                         sync_batchnorm=True,
                         log_every_n_steps=args.log_steps,
                         precision=args.precision,
                         replace_sampler_ddp=False)
    trainer.fit(model, data)


def main():
    args = get_arguments()
    print(args)

    train_wasrt(args)


if __name__ == '__main__':
    main()
