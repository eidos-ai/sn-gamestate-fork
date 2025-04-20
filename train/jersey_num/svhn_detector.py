from abc import ABC, abstractmethod
from ultralytics import YOLO
import numpy as np
import requests
import torch
from PIL import Image
from transformers import LlamaForConditionalGeneration, AutoProcessor
import torchvision.transforms as transforms
from typing import Optional, Tuple, Union

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

class LlamaNumberDet(NumberDetector):
    """
    Number detector using LLama-3 Vision model for binary classification (presence/absence of digits).
    """

    def __init__(self, model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"):
        """
        Initializes the LLama-3 Vision detector.

        Parameters:
            model_id: HuggingFace model identifier for LLama-3 Vision
        """
        self.model_id = model_id
        self.model = LlamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.tie_weights()
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        self.system_prompt = '''{
            "system": "You are a vision assistant that detects digits in images.",
            "instructions": [
                "Respond only in JSON format with {'digits': boolean}",
                "True if digits are present, False otherwise",
                "No additional commentary"
            ]
        }'''

    def detect(self, image: Union[np.ndarray, Image.Image]) -> bool:
        """
        Detects presence of digits in an image using LLama-3 Vision.

        Parameters:
            image: Input image as either:
                  - numpy.ndarray: Shape (H, W, 3), values 0-255
                  - PIL.Image: RGB image

        Returns:
            bool: True if digits detected, False otherwise
        """
        user_prompt = "<|image|>\nAre there digits in this image? Respond in JSON format."
        full_prompt = f"SYSTEM: {self.system_prompt}\nUSER: {user_prompt}\nASSISTANT:"
        
        inputs = self.processor(
            text=full_prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(**inputs, max_new_tokens=50)
        response = self.processor.decode(output[0], skip_special_tokens=True)
        
        # Extract JSON response
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            result = json.loads(json_str)
            return result.get('digits', False)
        except (ValueError, KeyError):
            return False

class SVHNNumberDet(NumberDetector):
    """
    Digit detector using YOLO model trained on SVHN dataset.
    """

    def __init__(self, model_path: str, threshold: float = 0.5):
        """
        Initializes the SVHN detector.

        Parameters:
            model_path: Path to YOLO model weights (.pt file)
            threshold: Confidence threshold (0-1) for detection
        """
        self.model = YOLO(model_path)
        self.threshold = threshold

    def detect(self, image: Union[np.ndarray, Image.Image]) -> bool:
        """
        Detects digits in an image using SVHN-trained YOLO model.

        Parameters:
            image: Input image as either:
                  - numpy.ndarray: Shape (H, W, 3), values 0-255
                  - PIL.Image: RGB image

        Returns:
            bool: True if digits detected above confidence threshold
        """
        detections = self.model(image, verbose=False)
        scores = detections[0].boxes.conf.cpu().numpy()
        return any(score > self.threshold for score in scores)

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