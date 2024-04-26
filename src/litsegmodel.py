import torch
import pytorch_lightning as pl
from argparse import Namespace
import yaml

import segmentation_models_pytorch as smp
from torchmetrics import JaccardIndex, ConfusionMatrix


with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    
LR = config['LR']
T_0 = config['T_0']
OPTIMIZER = config['OPTIMIZER']
LSMOOTH = config['LSMOOTH']


class LitSegModel(pl.LightningModule):
    def __init__(self, model,):
        super().__init__()
        self.model = model
        optimizer = OPTIMIZER
        self.optimizer_class = getattr(torch.optim, optimizer)
        self.model = model
        self.lr = LR
        self.T_0 = T_0
        self.jaccard = JaccardIndex(task="multiclass", num_classes=2)
        self.confmat = ConfusionMatrix(task="binary", num_classes=2)
        self.loss_fn = smp.losses.SoftCrossEntropyLoss(smooth_factor=LSMOOTH)

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.model(x)
        return embedding

    def predict(self, x):
        self.model.eval()
        torch.set_grad_enabled(False)

        logits = self.model(x)
        return torch.argmax(logits, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze()
        x = x.squeeze()
        logits = self(x)
        # Log batch loss
        #loss = torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=LSMOOTH)(logits, y).mean()
        loss = self.loss_fn(logits, y)
        #log jaccard index
        jaccard = self.jaccard(torch.argmax(logits, dim=1).float(), y)
        
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_jaccard",
            jaccard,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        outputs = {"loss": loss, "jaccard": jaccard}

        return outputs

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        torch.set_grad_enabled(False)

        x, y = batch
        y = y.squeeze()
        x = x.squeeze()

        logits = self(x)
        # Log batch loss
        #logits = torch.argmax(logits, dim=1).float()
        #loss = torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=LSMOOTH)(logits, y).mean()
        loss = self.loss_fn(logits, y)
        #log jaccard index
        jaccard = self.jaccard(torch.argmax(logits, dim=1).float(), y)
        
        self.log(
            "valid_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )        
        
        self.log(
            "valid_jaccard",
            jaccard,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        outputs = {"loss": loss, "jaccard": jaccard}

        return outputs

    def test_step(self, batch, batch_idx):
        self.model.eval()
        torch.set_grad_enabled(False)

        x, y = batch
        y = y.squeeze()
        x = x.squeeze()

        logits = self(x)
        # Log batch loss
        #logits = torch.argmax(logits, dim=1).float()
        #loss = torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=LSMOOTH)(logits, y).mean()
        loss = self.loss_fn(logits, y)
        #log jaccard index
        jaccard = self.jaccard(torch.argmax(logits, dim=1).float(), y)
        #log confusion matrix
        confusion = self.confmat(torch.argmax(logits, dim=1).float(), y)
        
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "test_jaccard",
            jaccard,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        outputs = {"loss": loss, "jaccard": jaccard, "confusion": confusion}
        
        return outputs

    def configure_optimizers(self):
        parameters = filter(lambda x: x.requires_grad, self.parameters())
        optimizer = self.optimizer_class(parameters, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer, T_0=self.T_0
        )
        # return {
        #    "optimizer": optimizer,
        #    "scheduler": [scheduler],
        #    "monitor": "val/loss",
        # }
        return [optimizer], [scheduler]
