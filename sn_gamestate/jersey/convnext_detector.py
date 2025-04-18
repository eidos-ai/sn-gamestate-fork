from abc import ABC, abstractmethod
from ultralytics import YOLO
import numpy as np
import requests
import torch
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, Tuple, Union
import torch
import pytorch_lightning as pl
from typing import List, Tuple
from dataclasses import dataclass
from .models.convnext import ConvNext
from lightning.pytorch.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

from dataclasses import dataclass
@dataclass
class DatasetConfiguration:
    train_positive: Path  # Path to training positive samples
    train_negative: Path  # Path to training negative samples
    test_positive: Path   # Path to test positive samples
    test_negative: Path   # Path to test negative samples

#The dataset config won’t be used but needs to be declared    
config = DatasetConfiguration(
    train_positive=Path("/path/to/train/positive"),
    train_negative=Path("/path/to/train/negative"),
    test_positive=Path("/path/to/test/positive"),
    test_negative=Path("/path/to/test/negative")
)
import sys
sys.modules['__main__'].DatasetConfiguration = DatasetConfiguration

class BinaryJerseyDataset(Dataset):
    def __init__(self, positive_dir: Path, negative_dir: Path, transform=None):
        """
        Args:
            positive_dir: Path to directory with positive samples
            negative_dir: Path to directory with negative samples
            transform: Optional transform to be applied
        """
        self.positive_samples = None
        self.negative_samples = None
        self.transform = transform
        # self.all_samples = self.positive_samples + self.negative_samples
        # self.labels = [1] * len(self.positive_samples) + [0] * len(self.negative_samples)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        img_path = self.all_samples[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class BinaryJerseyModule(pl.LightningModule):
    def __init__(self, data_conf: DatasetConfiguration=config, model_name="convnextv2_nano.fcmae_ft_in1k", 
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

class NumberDetector(ABC):
    """
    Abstract base class for detecting numbers in images.
    """

    @abstractmethod
    def detect(self, image: Union[np.ndarray, Image.Image]) -> bool:
        """
        Detects numbers in an image.

        Parameters:
            image: Input image as either:
                  - numpy.ndarray: RGB image with shape (H, W, 3) and values in [0, 255]
                  - PIL.Image: RGB image object

        Returns:
            bool: True if numbers are detected, False otherwise
        """
        pass
    
class ConvNext2NumberDet(NumberDetector):
    """
    Binary classifier using ConvNextV2 architecture for digit detection.
    """

    def __init__(self, model_path: str, threshold: float = 0.5):
        """
        Initializes the ConvNextV2 detector.

        Parameters:
            model_path: Path to model checkpoint
            threshold: Classification threshold (0-1)
        """
        self.model = BinaryJerseyModule.load_from_checkpoint(model_path)
        self.threshold = threshold
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def detect(self, image: Union[np.ndarray, Image.Image]) -> bool:
        """
        Classifies whether an image contains digits using ConvNextV2.

        Parameters:
            image: Input image as either:
                  - numpy.ndarray: Shape (H, W, 3), values 0-255
                  - PIL.Image: RGB image

        Returns:
            bool: True if classified as containing digits
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        probability = torch.sigmoid(logits).item()
        return probability > self.threshold