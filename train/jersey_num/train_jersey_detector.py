from pathlib import Path
import os

# Set environment variables in the notebook
os.environ["MLFLOW_TRACKING_USERNAME"] = "SoccerNetGR2025"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "soccer2025"

from torch.utils.data import DataLoader, Dataset
import torch
import pytorch_lightning as pl
from typing import List, Tuple
import torchvision.transforms as transforms
from PIL import Image
from dataclasses import dataclass
from models.convnext import ConvNext
from lightning.pytorch.loggers import MLFlowLogger

class BinaryJerseyDataset(Dataset):
    def __init__(self, positive_dir: Path, negative_dir: Path, transform=None):
        """
        Args:
            positive_dir: Path to directory with positive samples
            negative_dir: Path to directory with negative samples
            transform: Optional transform to be applied
        """
        self.positive_samples = list(positive_dir.glob('*'))
        self.negative_samples = list(negative_dir.glob('*'))
        self.transform = transform
        self.all_samples = self.positive_samples + self.negative_samples
        self.labels = [1] * len(self.positive_samples) + [0] * len(self.negative_samples)

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        img_path = self.all_samples[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

@dataclass
class DatasetConfiguration:
    train_positive: Path  # Path to training positive samples
    train_negative: Path  # Path to training negative samples
    test_positive: Path   # Path to test positive samples
    test_negative: Path   # Path to test negative samples

class BinaryJerseyTrainer(pl.LightningModule):
    def __init__(self, data_conf: DatasetConfiguration, model_name="convnextv2_nano.fcmae_ft_in1k", 
                 batch_size=32, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        
        # Binary classification model (output size = 1)
        self.model = ConvNext(model_name, 1, learning_rate)  # Modified for binary output
        
        # Transforms
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Datasets
        self.train_dataset = BinaryJerseyDataset(
            data_conf.train_positive, 
            data_conf.train_negative,
            self.train_transform
        )
        
        self.test_dataset = BinaryJerseyDataset(
            data_conf.test_positive,
            data_conf.test_negative,
            self.val_transform
        )
        
        self.val_dataset = BinaryJerseyDataset(
            data_conf.test_positive,
            data_conf.test_negative,
            self.val_transform
        )
        
        # Data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=19)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, num_workers=19)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, num_workers=19)

    def val_dataloader(self):
        # Return the validation data loader
        return self.val_loader
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y.float().unsqueeze(1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y.float().unsqueeze(1))
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        # Calculate accuracy
        preds = torch.sigmoid(y_hat) > 0.5
        acc = (preds == y.unsqueeze(1)).float().mean()
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y.float().unsqueeze(1))
        self.log('test_loss', loss)
        
        preds = torch.sigmoid(y_hat) > 0.5
        acc = (preds == y.unsqueeze(1)).float().mean()
        self.log('test_acc', acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

def train_binary(train_pos, train_neg, test_pos, test_neg, 
                epochs=10, batch_size=32, model_name="convnextv2_nano.fcmae_ft_in1k"):
    
    data_config = DatasetConfiguration(
        train_positive=Path(train_pos),
        train_negative=Path(train_neg),
        test_positive=Path(test_pos),
        test_negative=Path(test_neg)
    )
    
    model = BinaryJerseyTrainer(data_config, model_name=model_name, batch_size=batch_size)
    
    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",         # Monitor validation loss (or another metric)
        save_top_k=1,               # Save the best model
        mode="min",                 # Save the checkpoint when validation loss is minimized
        dirpath=Path("./saved_models"),   # Directory to save the checkpoint
        filename=f"best_model_{model_name}",      # Checkpoint file name
    )
    
    mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="http://trackme.eidos.ai/")
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        log_every_n_steps=10,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        logger=mlf_logger,
        default_root_dir=Path("./saved_models")  # Custom save directory
    )
    
    trainer.fit(model, model.train_loader)
    trainer.test(model, model.test_loader)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--model_name', type=str, default="convnextv2_nano.fcmae_ft_in1k", required=False, help='ConNext Model to use')
    
    args = vars(parser.parse_args())
    
    train_binary(
        "./mixed_dataset/train/jerseys",
        "./mixed_dataset/train/no-number",
        "./mixed_dataset/test/jerseys",
        "./mixed_dataset/test/no-number",
        epochs = 40,
        model_name = args["model_name"]
    )