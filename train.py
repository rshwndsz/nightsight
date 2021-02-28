import math
import os
import glob
from getpass import getpass

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F
import torchvision as tv
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold, train_test_split
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from nightsight.log import logger
from nightsight import utils
from nightsight import loss
from nightsight import data
from nightsight import model

CONSTANTS = {
    'SEED': 81,
}


class FinalNet(pl.LightningModule):
    def __init__(self, hparams):
        super(FinalNet, self).__init__()
        self.hparams = hparams
        self.enhance_net = model.EnhanceNetNoPool().apply(utils.weights_init)

    def forward(self, x):
        return self.enhance_net(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.hparams['lr'],
                                weight_decay=self.hparams['weight_decay'])

    def loss_function(self, inputs, enhanced_images, A):
        L_color = loss.LColorLoss()
        L_spa = loss.LSpaLoss(type_ref=inputs)
        L_exp = loss.LExpLoss(16, 0.6)
        L_tv = loss.LTVLoss()

        loss_tv = L_tv(A)
        loss_spa = torch.mean(L_spa(enhanced_images, inputs))
        loss_col = torch.mean(L_color(enhanced_images))
        loss_exp = torch.mean(L_exp(enhanced_images))

        loss = loss_spa + 10 * loss_exp + 5 * loss_col + 200 * loss_tv
        return loss

    def prepare_data(self):
        train_transform = A.Compose([
            A.VerticalFlip(p=0.1),
            A.HorizontalFlip(p=0.6),
            A.Resize(self.hparams['image_size'],
                     self.hparams['image_size'],
                     interpolation=4,
                     p=1),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), p=1),
            ToTensorV2(),
        ])
        test_transform = A.Compose([
            A.Resize(self.hparams['image_size'],
                     self.hparams['image_size'],
                     interpolation=4,
                     p=1),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), p=1),
            ToTensorV2(),
        ])
        self.train_ds = ZeroDceDS("data/train_data/",
                                  "*.jpg",
                                  train=True,
                                  transform=train_transform)
        self.val_ds = ZeroDceDS("data/test_data/DICM",
                                "*.jpg",
                                train=False,
                                transform=test_transform)
        self.test_ds = ZeroDceDS("data/test_data/LIME",
                                 "*.bmp",
                                 train=False,
                                 transform=test_transform)

    def train_dataloader(self):
        return D.DataLoader(self.train_ds,
                            batch_size=self.hparams['batch_size']['train'],
                            num_workers=4,
                            pin_memory=True,
                            shuffle=True)

    def val_dataloader(self):
        return D.DataLoader(self.val_ds,
                            batch_size=self.hparams['batch_size']['val'],
                            num_workers=4,
                            pin_memory=True,
                            shuffle=False)

    def test_dataloader(self):
        return D.DataLoader(self.test_ds,
                            batch_size=self.hparams['batch_size']['test'],
                            num_workers=4,
                            pin_memory=True,
                            shuffle=False)

    def training_step(self, batch, batch_idx):
        images = batch
        enhanced_1s, enhanced, A = self(images)
        loss = self.loss_function(images, enhanced, A)
        self.logger.experiment.log_metric('step_train_loss', loss)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.log_metric('epoch_train_loss', avg_loss)

    def validation_step(self, batch, batch_idx):
        images = batch
        enhanced_1s, enhanced, A = self(images)
        loss = self.loss_function(images, enhanced, A)

        self.logger.experiment.log_metric('step_val_loss', loss)
        if batch_idx < 4:
            display_images = torch.cat([clean, noisy, recon])
            self.logger.experiment.log_image(
                f'val_results_b{batch_idx}',
                make_grid(display_images, nrow=display_images.size(0) // 3))

        return {
            'val_loss': loss,
            'input_images': images,
            'enhanced_images': enhanced
        }

    def validation_epoch_end(self, outputs):
        # Compute avg_val_loss from previous results
        avg_val_loss = torch.stack([output['val_loss']
                                    for output in outputs]).mean()
        # Log avg_val_loss to use for `early stopping` and `model_checkpoint`
        self.log('avg_val_loss', avg_val_loss)
        # Log avg_val_loss to the experiment for training viz
        self.logger.experiment.log_metric('epoch_val_loss', avg_val_loss)

    def test_step(self, batch, batch_idx):
        images = batch
        enhanced_1s, enhanced, A = self(images)
        log_tensor(self.logger, f'test_results', torch.cat([images, enhanced]))


def post_train(trainer):
    # Log model summary
    for chunk in [x for x in str(model).split('\n')]:
        neptune_logger.experiment.log_text('model_summary', str(chunk))

    # Which GPUs where used?
    gpu_list = [
        f'{i}:{torch.cuda.get_device_name(i)}'
        for i in range(torch.cuda.device_count())
    ]
    neptune_logger.experiment.log_text('GPUs used', ', '.join(gpu_list))

    # Log best 3 model checkpoints to Neptune
    for k in model_checkpoint.best_k_models.keys():
        model_name = 'checkpoints/' + k.split('/')[-1]
        neptune_logger.experiment.log_artifact(k, model_name)

    # Save model at the last epoch
    last_model_path = f"zero-dce/checkpoints/epoch={trainer.current_epoch}.ckpt"
    trainer.save_checkpoint(last_model_path)
    neptune_logger.experiment.log_artifact(
        last_model_path, 'checkpoints/' + last_model_path.split('/')[-1])

    # Log score of the best model checkpoint
    neptune_logger.experiment.set_property(
        'best_model_score', model_checkpoint.best_model_score.tolist())

    # Testing
    trainer.test()


if __name__ == "__main__":
    # Setup
    logger.setLevel(logging.INFO)
    pl.seed_everything(CONSTANTS['SEED'])

    api_token = getpass("Enter Neptune.ai API token: ")
    neptune_logger = NeptuneLogger(
        api_key=api_token,
        project_name="rshwndsz/nightsight",
        close_after_fit=False,
        experiment_name="zero-dce",
        params=hparams,
        tags=["pytorch-lightning", "low-light-enhancement"]
    )

    # Config
    hparams = {
        'lr': 0.0001,
        'weight_decay': 0.0001,
        'batch_size': { 'train': 8, 'val': 4, 'test': 4 },
        'image_size': 256,
        'gradient_clip_val': 0.1,
        'max_epochs': 200,
        'min_epochs': 10,
        'check_val_every_n_epoch': 4,
        'precision': 16,
        'benchmark': True,
        'deterministic': False,
        'use_gpu': torch.cuda.is_available(),
    }

    # Callbacks
    model_checkpoint = ModelCheckpoint(
        filepath='checkpoints/{epoch:03d}-{epoch_val_loss:.2f}',
        save_top_k=5,
        monitor='val_loss',
        mode='min',
        period=5
    )
    early_stop_callback = EarlyStopping(monitor='epoch_val_loss',
                                        min_delta=0.00,
                                        patience=3,
                                        verbose=True,
                                        mode='min'
    )

    # Big bois
    model = FinalNet(hparams=hparams)
    trainer = pl.Trainer(
        gpus=-1 if hparams['use_gpu'] else 0,
        precision=hparams['precision'],
        gradient_clip_val=hparams['gradient_clip_val'],
        benchmark=hparams['benchmark'],
        deterministic=hparams['deterministic'],
        max_epochs=hparams['max_epochs'],
        min_epochs=hparams['min_epochs'],
        check_val_every_n_epoch=hparams['check_val_every_n_epoch'],
        logger=neptune_logger,
        checkpoint_callback=model_checkpoint,
        callbacks=[early_stop_callback],
    )

    # Train!
    trainer.fit(model)
    post_train(trainer)

    # Stop experiment
    neptune_logger.experiment.stop()
