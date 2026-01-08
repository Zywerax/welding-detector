"""
ML Classification Service - Wykrywanie defektów spawów
EfficientNet-B0 + Grad-CAM dla wizualizacji uwagi modelu
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

logger = logging.getLogger(__name__)


class WeldDataset(Dataset):
    """Dataset dla zdjęć spawów OK/NOK"""
    
    def __init__(self, data_dir: Path, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.classes = ['ok', 'nok']
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((img_path, class_idx))
                for img_path in class_dir.glob("*.png"):
                    self.samples.append((img_path, class_idx))
        
        logger.info(f"Loaded {len(self.samples)} samples from {data_dir}")
    
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


class GradCAM:
    """Grad-CAM dla wizualizacji obszarów uwagi"""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Generuj heatmapę Grad-CAM"""
        self.model.eval()
        
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        if self.gradients is None:
            raise RuntimeError("Gradients not captured. Check target layer registration.")
        
        if self.activations is None:
            raise RuntimeError("Activations not captured. Check target layer registration.")
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        
        for i in range(activations.shape[0]):
            activations[i] *= pooled_gradients[i]
        
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        
        return heatmap


class MLClassificationService:
    """Serwis do klasyfikacji spawów OK/NOK z Grad-CAM"""
    
    def __init__(self, models_dir: str = "models", labels_dir: str = "labels"):
        self.models_dir = Path(models_dir)
        self.labels_dir = Path(labels_dir)
        self.training_data_dir = self.labels_dir / "training_data"
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.grad_cam = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.training_info_path = self.models_dir / "training_info.json"
        
        logger.info(f"ML Service initialized. Device: {self.device}")
        
        # Automatycznie załaduj ostatni model jeśli istnieje
        if self.load_model("latest_model.pth"):
            logger.info("✅ Loaded latest model successfully")
        else:
            logger.info("ℹ️ No pretrained model found, will need to train first")
    
    def get_training_data_stats(self) -> Dict[str, Any]:
        """Pobierz statystyki danych treningowych"""
        ok_dir = self.training_data_dir / "ok"
        nok_dir = self.training_data_dir / "nok"
        
        ok_count = len(list(ok_dir.glob("*.jpg"))) + len(list(ok_dir.glob("*.png"))) if ok_dir.exists() else 0
        nok_count = len(list(nok_dir.glob("*.jpg"))) + len(list(nok_dir.glob("*.png"))) if nok_dir.exists() else 0
        
        return {
            "ok_samples": ok_count,
            "nok_samples": nok_count,
            "total_samples": ok_count + nok_count,
            "min_samples_per_class": min(ok_count, nok_count),
            "ready_for_training": min(ok_count, nok_count) >= 20,
            "class_balance": round(min(ok_count, nok_count) / max(ok_count, nok_count, 1), 2)
        }
    
    def create_model(self) -> nn.Module:
        """Stwórz model EfficientNet-B0 dla klasyfikacji binarnej"""
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
        return model.to(self.device)
    
    def train(
        self,
        epochs: int = 20,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        augment: bool = True
    ) -> Dict[str, Any]:
        """Trenuj model na zebranych danych"""
        
        stats = self.get_training_data_stats()
        if not stats["ready_for_training"]:
            raise ValueError(f"Insufficient training data. Need at least 20 samples per class. "
                           f"Current: OK={stats['ok_samples']}, NOK={stats['nok_samples']}")
        
        logger.info(f"Starting training with {stats['total_samples']} samples")
        
        # Transformacje
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(10) if augment else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(brightness=0.2, contrast=0.2) if augment else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        full_dataset = WeldDataset(self.training_data_dir, transform=train_transform)
        
        val_size = int(len(full_dataset) * validation_split)
        train_size = len(full_dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        self.model = self.create_model()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "best_val_acc": 0,
            "best_epoch": 0
        }
        
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
                self._save_model("best_model.pth")
        
        self._save_model("latest_model.pth")
        
        training_info = {
            "trained_at": datetime.now().isoformat(),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_samples": train_size,
            "val_samples": val_size,
            "best_val_acc": best_val_acc,
            "best_epoch": history["best_epoch"],
            "device": str(self.device)
        }
        
        with open(self.training_info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
        
        self._setup_gradcam()
        
        return history
    
    def _save_model(self, filename: str):
        """Zapisz model PyTorch"""
        if self.model:
            path = self.models_dir / filename
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'num_classes': 2,
                'model_name': 'efficientnet_b0'
            }, path)
            logger.info(f"Model saved to {path}")
    
    def load_model(self, filename: str = "latest_model.pth") -> bool:
        """Wczytaj model"""
        path = self.models_dir / filename
        if not path.exists():
            logger.warning(f"Model not found: {path}")
            return False
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model = self.create_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self._setup_gradcam()
            
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _setup_gradcam(self):
        """Zainicjuj Grad-CAM z ostatnią warstwą konwolucyjną"""
        if self.model:
            target_layer = self.model.conv_head if hasattr(self.model, 'conv_head') else self.model.features[-1]
            self.grad_cam = GradCAM(self.model, target_layer)
            logger.info("Grad-CAM initialized")
    
    def predict(self, image: np.ndarray, with_gradcam: bool = True) -> Dict[str, Any]:
        """Wykonaj predykcję na obrazie"""
        if not self.model:
            if not self.load_model():
                raise RuntimeError("No model available. Train first or load existing model.")
        
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        
        # Konwertuj do RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.set_grad_enabled(with_gradcam):
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            confidence, predicted = probabilities.max(1)
            predicted_class = "ok" if predicted.item() == 0 else "nok"
        
        result = {
            "prediction": predicted_class,
            "confidence": round(confidence.item() * 100, 2),
            "class_probabilities": {
                "ok": round(probabilities[0, 0].item() * 100, 2),
                "nok": round(probabilities[0, 1].item() * 100, 2)
            },
            "gradcam_heatmap": None
        }
        
        if with_gradcam and self.grad_cam:
            input_tensor.requires_grad = True
            heatmap = self.grad_cam.generate(input_tensor, target_class=int(predicted.item()))
            heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
            result["gradcam_heatmap"] = heatmap_resized
        
        return result
    
    def create_gradcam_overlay(
        self, 
        image: np.ndarray, 
        heatmap: np.ndarray, 
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """Stwórz overlay obrazu z heatmapą Grad-CAM"""
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(np.ascontiguousarray(heatmap_uint8, dtype=np.uint8), colormap)
        
        if heatmap_colored.shape[:2] != image.shape[:2]:
            heatmap_colored = cv2.resize(heatmap_colored, (image.shape[1], image.shape[0]))
        
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay
    
    def get_model_info(self) -> Dict[str, Any]:
        """Pobierz informacje o modelu"""
        info = {
            "model_loaded": self.model is not None,
            "device": str(self.device),
            "gradcam_available": self.grad_cam is not None,
            "training_info": None
        }
        
        if self.training_info_path.exists():
            with open(self.training_info_path, 'r') as f:
                info["training_info"] = json.load(f)
        
        info["available_models"] = [f.name for f in self.models_dir.glob("*.pth")]
        
        return info


# Singleton instance
_service_instance = None

def get_ml_service() -> MLClassificationService:
    """Pobierz singleton MLClassificationService"""
    global _service_instance
    if _service_instance is None:
        _service_instance = MLClassificationService()
    return _service_instance
