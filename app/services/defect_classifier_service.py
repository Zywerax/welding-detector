"""
Defect Classifier Service - calssify welding defects into 9 categories using EfficientNet-B0
Extension ML Classification Service to handle defect-specific logic, dataset, and model architecture. 
Provides training and prediction capabilities with Grad-CAM visualization.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import timm

from app.services.ml_classification_service import MLClassificationService

logger = logging.getLogger(__name__)


# Types of welding defects we want to classify
DEFECT_CLASSES = [
    'porosity',        # 0
    'crack',           # 1
    'lack_of_fusion',  # 2
    'undercut',        # 3
    'burn_through',    # 4
    'spatter',         # 5
    'irregular_bead',  # 6
    'contamination',   # 7
    'other'            # 8
]


class DefectDataset(Dataset):
    """Dataset for classifying types of welding defects"""
    
    def __init__(self, data_dir: Path, transform=None, available_classes=None):
        self.data_dir = Path(data_dir) / "defects"
        self.transform = transform
        self.samples = []
        
        # Use only available classes (with data)
        self.classes = available_classes if available_classes else DEFECT_CLASSES
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((img_path, class_idx))
                for img_path in class_dir.glob("*.png"):
                    self.samples.append((img_path, class_idx))
        
        logger.info(f"Loaded {len(self.samples)} defect samples from {self.data_dir} ({len(self.classes)} classes)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


class DefectClassifierService(MLClassificationService):
    """Service for classifying types of welding defects (9 classes)"""
    
    def __init__(self):
        # Do not call super().__init__ - initialize manually
        self.models_dir = Path("models/defects")
        self.labels_dir = Path("labels")
        self.training_data_dir = self.labels_dir / "training_data"
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.grad_cam = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.training_info_path = self.models_dir / "training_info.json"
        self.num_classes = None  # Will be set dynamically during training
        self.class_names = []  # Will be set dynamically during training
        
        logger.info(f"Defect Classifier Service initialized. Device: {self.device}")
        
        # Load defect model if it exists
        if self.load_model("defect_classifier.pth"):
            logger.info("Loaded defect classifier successfully")
        else:
            logger.info("ℹ️ No defect classifier found, will need to train first")
    
    def create_model(self) -> nn.Module:
        """Create EfficientNet-B0 model for defect classification"""
        if self.num_classes is None:
            raise ValueError("num_classes not set. Train the model first or load existing model.")
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=self.num_classes)
        return model.to(self.device)
    
    def get_training_data_stats(self) -> Dict[str, Any]:
        """Get training data statistics for defects"""
        defects_dir = self.training_data_dir / "defects"
        
        class_counts = {}
        total = 0
        
        for defect_type in DEFECT_CLASSES:
            defect_dir = defects_dir / defect_type
            count = 0
            if defect_dir.exists():
                count = len(list(defect_dir.glob("*.jpg"))) + len(list(defect_dir.glob("*.png")))
            class_counts[defect_type] = count
            total += count
        
        # Only classes with data
        available_classes = {k: v for k, v in class_counts.items() if v > 0}
        min_samples = min(available_classes.values()) if available_classes else 0
        num_available_classes = len(available_classes)
        
        return {
            "class_counts": class_counts,
            "available_classes": list(available_classes.keys()),
            "total_samples": total,
            "min_samples_per_class": min_samples,
            "ready_for_training": min_samples >= 10 and num_available_classes >= 2 and total >= 50,
            "num_classes": num_available_classes
        }
    
    def train(
        self,
        epochs: int = 30,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        augment: bool = True
    ) -> Dict[str, Any]:
        """Train defect classification model"""
        
        stats = self.get_training_data_stats()
        if not stats["ready_for_training"]:
            raise ValueError(f"Insufficient training data. Need at least 10 samples per class (with data), "
                           f"2+ classes, and 50 total. Current: {stats['class_counts']}")
        
        # Set classes dynamically
        self.class_names = stats['available_classes']
        self.num_classes = len(self.class_names)
        
        logger.info(f"Starting defect classification training with {stats['total_samples']} samples")
        
        # Use DefectDataset instead of WeldDataset
        from torchvision import transforms
        from torch.utils.data import DataLoader
        
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(15) if augment else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(brightness=0.3, contrast=0.3) if augment else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        full_dataset = DefectDataset(self.training_data_dir, transform=train_transform, available_classes=self.class_names)
        
        val_size = int(len(full_dataset) * validation_split)
        train_size = len(full_dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        self.model = self.create_model()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "best_val_acc": 0, "best_epoch": 0}
        best_val_acc = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_acc = 100. * train_correct / train_total
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_acc = 100. * val_correct / max(val_total, 1)
            val_loss /= max(len(val_loader), 1)
            
            scheduler.step(val_loss)
            
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                history["best_val_acc"] = val_acc
                history["best_epoch"] = epoch + 1
                self._save_model("defect_classifier_best.pth")
        
        self._save_model("defect_classifier.pth")
        
        # save training info
        import json
        from datetime import datetime
        
        training_info = {
            "trained_at": datetime.now().isoformat(),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_samples": train_size,
            "val_samples": val_size,
            "best_val_acc": best_val_acc,
            "best_epoch": history["best_epoch"],
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "device": str(self.device)
        }
        
        with open(self.training_info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
        
        self._setup_gradcam()
        
        return history
    
    def predict(self, image, with_gradcam: bool = True) -> Dict[str, Any]:
        """Perform defect type prediction"""
        if self.model is None:
            if not self.load_model("defect_classifier.pth"):
                raise RuntimeError("No defect classifier available. Train first.")
        
        if self.model is None:
            raise RuntimeError("Failed to load or initialize model for prediction.")
        
        import numpy as np
        
        # Convert to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        transformed = self.transform(image)
        if not isinstance(transformed, torch.Tensor):
            from torchvision.transforms import ToTensor
            transformed = ToTensor()(transformed)
        input_tensor = transformed.unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.set_grad_enabled(with_gradcam):
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            confidence, predicted = probabilities.max(1)
            predicted_class = self.class_names[int(predicted.item())]
        
        result = {
            "prediction": predicted_class,
            "confidence": round(confidence.item() * 100, 2),
            "class_probabilities": {
                name: round(probabilities[0, i].item() * 100, 2)
                for i, name in enumerate(self.class_names)
            },
            "gradcam_heatmap": None
        }
        
        if with_gradcam and self.grad_cam:
            input_tensor.requires_grad = True
            heatmap = self.grad_cam.generate(input_tensor, target_class=int(predicted.item()))
            heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
            result["gradcam_heatmap"] = heatmap_resized
        
        return result
    
    def _save_model(self, filename: str):
        """Save defect classification model"""
        if self.model:
            path = self.models_dir / filename
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'num_classes': self.num_classes,
                'class_names': self.class_names,
                'model_name': 'efficientnet_b0'
            }, path)
            logger.info(f"Defect classifier saved to {path}")
    
    def load_model(self, filename: str = "defect_classifier.pth") -> bool:
        """Load defect classification model"""
        path = self.models_dir / filename
        if not path.exists():
            logger.warning(f"Defect classifier not found: {path}")
            return False
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            # Load saved classes and number of classes
            self.class_names = checkpoint.get('class_names', DEFECT_CLASSES)
            self.num_classes = checkpoint.get('num_classes', len(self.class_names))
            self.model = self.create_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self._setup_gradcam()
            
            logger.info(f"Defect classifier loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load defect classifier: {e}")
            return False
    
    def _setup_gradcam(self):
        """Initialize Grad-CAM with the last convolutional layer"""
        if self.model:
            from app.services.ml_classification_service import GradCAM
            # For EfficientNet models, use conv_head or the last layer in features
            target_layer: nn.Module
            if hasattr(self.model, 'conv_head') and isinstance(self.model.conv_head, nn.Module):
                target_layer = self.model.conv_head
            elif hasattr(self.model, 'features') and isinstance(self.model.features, nn.Module):
                target_layer = self.model.features
            else:
                # Fallback to the model itself
                target_layer = self.model
            self.grad_cam = GradCAM(self.model, target_layer)
            logger.info("Grad-CAM initialized for defect classifier")


# Singleton instance
_defect_service_instance = None

def get_defect_classifier_service() -> DefectClassifierService:
    """Get singleton DefectClassifierService"""
    global _defect_service_instance
    if _defect_service_instance is None:
        _defect_service_instance = DefectClassifierService()
    return _defect_service_instance
