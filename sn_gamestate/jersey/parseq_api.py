import argparse
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args
from tracklab.utils.collate import default_collate, Unbatchable
from tracklab.pipeline.detectionlevel_module import DetectionLevelModule

log = logging.getLogger(__name__)
CHECKPOINT_PATH = Path("/home/federico/parseq/outputs/parseq/with_wd0.1/checkpoints/epoch=12-step=377-val_accuracy=65.0406-val_NED=74.8347.ckpt")


class PARSEQ(DetectionLevelModule):
    """A detection-level module for recognizing jersey numbers using PARSEQ model.
    
    Attributes:
        input_columns: List of input column names required for processing.
        output_columns: List of output column names produced by the module.
        collate_fn: Function to collate data into batches.
    """
    input_columns = ["bbox_ltwh"]
    output_columns = ["jersey_number_detection", "jersey_number_confidence"]
    collate_fn = default_collate

    def __init__(self, batch_size: int, device: torch.device, 
                 checkpoint_path: Path = CHECKPOINT_PATH, 
                 tracking_dataset: Optional[Any] = None):
        """Initialize the PARSEQ jersey number recognition module.
        
        Args:
            batch_size: Number of samples per batch.
            device: The device (CPU/GPU) to run the model on.
            checkpoint_path: Path to the PARSEQ model checkpoint.
            tracking_dataset: Optional tracking dataset reference.
        """
        super().__init__(batch_size=batch_size)
        # Load PARSEQ model
        parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Update model with checkpoint weights
        if 'state_dict' in checkpoint:  # PyTorch Lightning checkpoint
            parseq.load_state_dict(checkpoint['state_dict'])
        else:  # Raw state_dict
            parseq.load_state_dict(checkpoint)
            
        self.model = parseq    
        self.device = device
        self.batch_size = batch_size
        self.model.eval()
        log.info(f"Tracking dataset: {tracking_dataset}")

    def no_jersey_number(self) -> Tuple[None, int]:
        """Return default values when no jersey number is detected."""
        return None, 0

    @torch.no_grad()
    def preprocess(self, image: np.ndarray, detection: pd.Series, 
                   metadata: pd.Series) -> Dict[str, Any]:
        """Preprocess the image crop for jersey number recognition.
        
        Args:
            image: The input image as a numpy array.
            detection: Pandas Series containing detection information.
            metadata: Pandas Series containing metadata about the image.
            
        Returns:
            A dictionary containing the preprocessed data.
        """
        l, t, r, b = detection.bbox.ltrb(
            image_shape=(image.shape[1], image.shape[0]), rounded=True
        )
        crop = image[t:b, l:r]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            crop = np.zeros((10, 10, 3), dtype=np.uint8)
        crop = Unbatchable([crop])
        return {"img": crop}

    @torch.no_grad()
    def process(self, batch: Dict[str, Any], detections: pd.DataFrame, 
                metadatas: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of images to recognize jersey numbers.
        
        Args:
            batch: Dictionary containing the batch data.
            detections: DataFrame containing detection information.
            metadatas: DataFrame containing metadata about the images.
            
        Returns:
            A DataFrame containing the recognition results.
        """
        images_np = [img.cpu().numpy() for img in batch['img']]
        del batch['img']
        return self.run_parseq_inference(images_np)
    
    @torch.no_grad()
    def run_parseq_inference(self, images_np: np.ndarray, 
                            save_dir: Path = Path('output_predictions')) -> pd.DataFrame:
        """Run PARSEQ inference on a list of images and save visualizations.
        
        Args:
            images_np: List of numpy array images to process.
            save_dir: Directory to save visualization results.
            
        Returns:
            DataFrame containing recognition results with columns:
            - jersey_number_detection: Recognized text
            - jersey_number_confidence: Confidence score
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "jersey_number_detection": [], 
            "jersey_number_confidence": []
        }

        for i, img in enumerate(images_np):
            pil_img = Image.fromarray(img, 'RGB')
            img_transform = SceneTextDataModule.get_transform(self.model.hparams.img_size)
            img_tensor = img_transform(pil_img).unsqueeze(0)

            # Run inference
            pred = self.model(img_tensor).softmax(-1)
            label, confidence = self.model.tokenizer.decode(pred)
            detected_text = label[0]
            conf_score = confidence[0].mean()

            # Draw prediction on image
            draw = ImageDraw.Draw(pil_img)
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except IOError:
                font = ImageFont.load_default()

            text = f"{detected_text} ({conf_score:.2f})"
            draw.text((10, 10), text, font=font, fill=(255, 0, 0))

            # Save visualization
            output_path = save_dir / f"pred_{i}.jpg"
            pil_img.save(output_path)

            # Store results
            result["jersey_number_detection"].append(detected_text)
            result["jersey_number_confidence"].append(conf_score)

        return pd.DataFrame(result)