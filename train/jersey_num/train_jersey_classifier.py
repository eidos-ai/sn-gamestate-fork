import os
from collections import defaultdict, Counter

# Set environment variables
os.environ["MLFLOW_TRACKING_USERNAME"] = "SoccerNetGR2025"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "soccer2025"

from torch.utils.data import DataLoader, Dataset
import torch
import pytorch_lightning as pl
from pathlib import Path
from typing import List, Tuple, Dict
import torchvision.transforms as transforms
from PIL import Image
from dataclasses import dataclass
from models.convnext import ConvNext
from lightning.pytorch.loggers import MLFlowLogger
from torchvision.transforms import functional as F
from PIL import ImageOps

class MultiClassJerseyDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, target_size=(224, 224)):
        """
        Args:
            root_dir: Directory with class folders (0-99)
            transform: Optional transform (applied AFTER padding/resizing)
            target_size: Model's expected input size (h, w)
        """
        self.root_dir = root_dir
        self.target_size = target_size
        self.transform = transform
        
        # Define all 100 classes (00-99)
        self.classes = [f"{i}" for i in range(100)]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}
        
        # Track missing classes and samples
        self.missing_classes = set()
        self.samples = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        
        # Scan directory structure
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                self.missing_classes.add(class_name)
                continue
                
            class_files = []
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith(valid_extensions) and not file.startswith('.'):
                        class_files.append(os.path.join(root, file))
            
            if not class_files:
                self.missing_classes.add(class_name)
                continue
                
            for file_path in class_files:
                self.samples.append((
                    file_path,
                    self.class_to_idx[class_name]
                ))

        # Print diagnostics
        present_classes = set(self.classes) - self.missing_classes
        print(f"Found {len(self.samples)} samples across {len(present_classes)} classes")
        print(f"Missing classes ({len(self.missing_classes)}): {sorted(self.missing_classes)}")
        
        # Calculate class distribution
        class_counts = Counter([label for _, label in self.samples])
        print("Class distribution:", class_counts.most_common())

    def _pad_to_square(self, img):
        """Add padding to make the image square"""
        w, h = img.size
        max_side = max(w, h)
        padding = (
            (max_side - w) // 2,  # Left
            (max_side - h) // 2,   # Top
            (max_side - w + 1) // 2,  # Right
            (max_side - h + 1) // 2   # Bottom
        )
        return ImageOps.expand(img, border=padding, fill=0)  # Black padding

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # 1. Pad to square
            img = self._pad_to_square(img)
            
            # 2. Resize to target dimensions
            img = F.resize(img, self.target_size)
            
            # 3. Apply transforms if specified
            if self.transform:
                img = self.transform(img)
                
            # Convert to tensor and normalize
            img = F.to_tensor(img)
            img = F.normalize(
                img, 
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return blank image if loading fails
            img = torch.zeros(3, *self.target_size)
            
        return img, label

@dataclass
class DatasetConfiguration:
    train_dir: str  # Path to training samples
    test_dir: str   # Path to test samples

class MultiClassJerseyTrainer(pl.LightningModule):
    def __init__(self, data_conf: DatasetConfiguration, model_name="convnextv2_nano.fcmae_ft_in1k", 
                 batch_size=32, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        
        # Initialize datasets first to discover classes
        self.train_dataset = MultiClassJerseyDataset(
            data_conf.train_dir,
            target_size=(224, 224),
            transform=transforms.Compose([
                #transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        )
        
        self.val_dataset = MultiClassJerseyDataset(
            data_conf.test_dir,
            target_size=(224, 224)
        )
        
        # Fixed 100-class output layer
        self.num_classes = 100
        self.model = ConvNext(model_name, self.num_classes, learning_rate=learning_rate)
        
        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=19,
            persistent_workers=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            num_workers=19,
            persistent_workers=True
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        
        # Calculate accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        # Calculate accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=3
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

def train_multiclass(train_dir, test_dir, 
                    epochs=40, batch_size=32, model_name="convnextv2_nano.fcmae_ft_in1k"):
    
    data_config = DatasetConfiguration(
        train_dir=train_dir,
        test_dir=test_dir
    )
    
    model = MultiClassJerseyTrainer(
        data_config, 
        model_name=model_name, 
        batch_size=batch_size
    )
    
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        save_top_k=1,
        mode="max",
        dirpath="./saved_models",
        filename=f"jersey_numbers_{model_name}_best",
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor="val_acc",
        patience=10,
        mode="max",
        verbose=True
    )
    
    mlf_logger = MLFlowLogger(
        experiment_name="jersey_number_recognition",
        tracking_uri="http://trackme.eidos.ai/"
    )
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        log_every_n_steps=20,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, early_stopping],
        logger=mlf_logger,
        default_root_dir="./saved_models",
        accelerator="auto",
        devices="auto"
    )
    
    trainer.fit(model, model.train_loader, model.val_loader)
    trainer.test(model, model.val_loader, ckpt_path="best")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train 100-class jersey number classifier')
    parser.add_argument('-m', '--model_name', type=str, default="convnextv2_nano.fcmae_ft_in1k", 
                       help='ConNext Model to use')
    parser.add_argument('-b', '--batch_size', type=int, default=32, 
                       help='Batch size for training')
    parser.add_argument('-e', '--epochs', type=int, default=40, 
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    train_multiclass(
        train_dir="./merged_dataset",  # Should contain subfolders 00, 01, ..., 99
        test_dir="./test_0.85",       # Should contain same subfolders
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_name=args.model_name
    )