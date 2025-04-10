import argparse
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

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

import os
DEFAULT_CHECKPOINT_PATH = Path(
    os.getenv("PARSEQ_CHECKPOINT_PATH", "/home/federico/parseq/outputs/parseq/with_wd0.1/checkpoints/epoch=12-step=377-val_accuracy=65.0406-val_NED=74.8347.ckpt") # the path is the default, if it is set as an env variable it will take the variable
)


class PARSEQ(DetectionLevelModule):
    """A detection-level module for recognizing jersey numbers using PARSEQ model.
    
    Attributes:
        input_columns: List of input column names required for processing.
        output_columns: List of output column names produced by the module.
        collate_fn: Function to collate data into batches.
    """
    input_columns = ["bbox_ltwh", "has_number"]
    output_columns = ["jersey_number_detection", "jersey_number_confidence"]
    collate_fn = default_collate

    def __init__(self, batch_size: int, device: torch.device, 
                 checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH, 
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
        """Preprocess the image crop for jersey number recognition."""
        l, t, r, b = detection.bbox.ltrb(
            image_shape=(image.shape[1], image.shape[0]), 
            rounded=True
        )
        crop = image[t:b, l:r]
        return {
            "img": Unbatchable([crop]),
            "has_number": detection.get('has_number', False)
        }

    @torch.no_grad()
    def process(self, batch: Dict[str, Any], detections: pd.DataFrame, 
                metadatas: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of images to recognize jersey numbers."""
        images = batch['img']
        has_numbers = batch['has_number']

        # Convert all images to numpy
        images_np = [img.cpu().numpy() for img in images]
        results = self.run_parseq_inference(images_np, has_numbers)
        detections["jersey_number_detection"] = results["jersey_number_detection"]
        detections["jersey_number_confidence"] = results["jersey_number_confidence"]
        return detections
        #return self.run_parseq_inference(images_np, has_numbers)

    @torch.no_grad()
    def run_parseq_inference(self, images_np: List[np.ndarray], has_numbers: List[bool],
                            save_dir: Path = Path('output_predictions_parseq')) -> pd.DataFrame:
        """Run PARSEQ inference on images."""
        save_dir.mkdir(parents=True, exist_ok=True)
        results = {
            "jersey_number_detection": [], 
            "jersey_number_confidence": []
        }

        for i, (img, has_number) in enumerate(zip(images_np, has_numbers)):
            if not has_number:
                results["jersey_number_detection"].append(0)
                results["jersey_number_confidence"].append(0.0)
                continue

            try:
                # Convert and transform image
                pil_img = Image.fromarray(img, 'RGB')
                img_transform = SceneTextDataModule.get_transform(self.model.hparams.img_size)
                img_tensor = img_transform(pil_img).unsqueeze(0)#.to(self.device)

                # Inference
                pred = self.model(img_tensor).softmax(-1)
                label, confidence = self.model.tokenizer.decode(pred)
                detected_text = label[0]
                conf_score = confidence[0].mean()
                print(f"Detected text: {detected_text}")
                print(f"confidence: {conf_score}")
                # Visualization
                draw = ImageDraw.Draw(pil_img)
                try:
                    font = ImageFont.truetype("arial.ttf", 24)
                except IOError:
                    font = ImageFont.load_default()
                draw.text((10, 10), f"{detected_text} ({conf_score:.2f})", 
                         font=font, fill="red")

                # Save
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = save_dir / f"pred_{timestamp}_{i}.jpg"
                pil_img.save(output_path)
                print(f"Saved visualization to {output_path}")

                results["jersey_number_detection"].append(detected_text)
                results["jersey_number_confidence"].append(float(conf_score))

            except Exception as e:
                print(f"Error processing image {i}: {e}")
                

        return pd.DataFrame(results)

#     @torch.no_grad()
#     def preprocess(self, image: np.ndarray, detection: pd.Series, 
#                    metadata: pd.Series) -> Dict[str, Any]:
#         """Preprocess the image crop for jersey number recognition.
        
#         Args:
#             image: The input image as a numpy array.
#             detection: Pandas Series containing detection information.
#             metadata: Pandas Series containing metadata about the image.
            
#         Returns:
#             A dictionary containing the preprocessed data.
#         """
#         l, t, r, b = detection.bbox.ltrb(
#             image_shape=(image.shape[1], image.shape[0]), rounded=True
#         )
#         has_number = detection.has_number
#         print(has_number)
#         crop = image[t:b, l:r]
#         # if crop.shape[0] == 0 or crop.shape[1] == 0:
#         #     crop = np.zeros((10, 10, 3), dtype=np.uint8)
#         crop = Unbatchable([crop])
#         return {"img": crop, "has_number": has_number}

#     @torch.no_grad()
#     def process(self, batch: Dict[str, Any], detections: pd.DataFrame, 
#                 metadatas: pd.DataFrame) -> pd.DataFrame:
#         """Process a batch of images to recognize jersey numbers.
        
#         Args:
#             batch: Dictionary containing the batch data.
#             detections: DataFrame containing detection information.
#             metadatas: DataFrame containing metadata about the images.
            
#         Returns:
#             A DataFrame containing the recognition results.
#         """
#         images = batch['img']
#         has_numbers = batch['has_number']
    
#         # Filter images where has_number is True and convert to numpy
#         images_np = [img.cpu().numpy() for img in images]
#         #images_np = [img.cpu().numpy() for img, has_num in zip(images, has_numbers) if has_num]

#         # Run inference only on relevant images
#         if images_np:
#             return self.run_parseq_inference(images_np, has_numbers)
#         else:
#             return pd.DataFrame()  # Return empty DataFrame if no images to process
    
    
#     @torch.no_grad()
#     def run_parseq_inference(self, images_np: np.ndarray, has_numbers,
#                             save_dir: Path = Path('output_predictions_parseq')) -> pd.DataFrame:
#         """Run PARSEQ inference on a list of images and save visualizations.
        
#         Args:
#             images_np: List of numpy array images to process.
#             save_dir: Directory to save visualization results.
            
#         Returns:
#             DataFrame containing recognition results with columns:
#             - jersey_number_detection: Recognized text
#             - jersey_number_confidence: Confidence score
#         """
#         save_dir.mkdir(parents=True, exist_ok=True)
#         result = {
#             "jersey_number_detection": [], 
#             "jersey_number_confidence": []
#         }

#         for i, img, has_number in enumerate(zip(images_np, has_numbers)):
#             if not has_number:
#                 result["jersey_number_detection"].append(0)
#                 result["jersey_number_confidence"].append(0.1)
                
#             pil_img = Image.fromarray(img, 'RGB')
#             img_transform = SceneTextDataModule.get_transform(self.model.hparams.img_size)
#             img_tensor = img_transform(pil_img).unsqueeze(0)

#             # Run inference
#             pred = self.model(img_tensor).softmax(-1)
#             label, confidence = self.model.tokenizer.decode(pred)
#             detected_text = label[0]
#             conf_score = confidence[0].mean()

#             # Draw prediction on image
#             draw = ImageDraw.Draw(pil_img)
#             try:
#                 font = ImageFont.truetype("arial.ttf", 24)
#             except IOError:
#                 font = ImageFont.load_default()

#             text = f"{detected_text} ({conf_score:.2f})"
#             draw.text((10, 10), text, font=font, fill=(255, 0, 0))

#             # Save visualization
#             output_path = save_dir / f"pred_{i}.jpg"
#             pil_img.save(output_path)

#             # Store results
#             result["jersey_number_detection"].append(detected_text)
#             result["jersey_number_confidence"].append(conf_score)

#         return pd.DataFrame(result)