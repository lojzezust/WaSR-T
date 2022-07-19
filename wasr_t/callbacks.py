import os
import torch

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only

class ModelExport(Callback):
    def __init__(self, output_dir):
        self.output_dir = output_dir

    @rank_zero_only
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        model = pl_module.model
        model_name = trainer.logger.name
        version = trainer.logger.version

        output_name = f'{model_name}_v{version}'
        output_dir = os.path.join(self.output_dir, output_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_path = os.path.join(output_dir, 'model.pth')
        torch.save(model, model_path)
        print(f'Exported to: {model_path}')
