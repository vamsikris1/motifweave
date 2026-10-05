import os
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class SareeDataset(Dataset):
    """
    Dataset loader for MotifWeave saree images.
    Loads from processed manifest or split CSV.
    """
    def __init__(self, csv_file, transform=None, label_encoder=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
        # Build deterministic mapping from source_id string to integer label
        if label_encoder is None:
            unique_sources = sorted(self.df['source_id'].unique())
            self.label_encoder = {src: idx for idx, src in enumerate(unique_sources)}
        else:
            self.label_encoder = label_encoder
            
        self.df['source_label'] = self.df['source_id'].map(self.label_encoder)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        
        # Open and ensure RGB
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            if self.transform:
                img_tensor = self.transform(img)
            else:
                img_tensor = T.ToTensor()(img)
                
        source_label = int(row['source_label'])
        source_id = str(row['source_id'])
        variant_type = str(row['variant_type'])
        
        return img_tensor, source_label, source_id, variant_type

def get_baseline_transforms(image_size=224, is_training=True):
    """
    Standard ImageNet transforms for baseline experiment (Experiment A).
    Explicitly excludes aggressive hue perturbation to establish a fair baseline.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    if is_training:
        return T.Compose([
            T.Resize((int(image_size * 1.14), int(image_size * 1.14))),
            T.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

def get_color_invariant_transforms(image_size=224, is_training=True, hue=0.30, saturation=0.20, brightness=0.15, contrast=0.15, p_grayscale=0.15):
    """
    Color-invariant transforms for Experiment B.
    Applies controlled chromatic perturbation and optional desaturation
    to decouple design motifs from color palettes while preserving spatial layout and weave geometry.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    if is_training:
        return T.Compose([
            T.Resize((int(image_size * 1.14), int(image_size * 1.14))),
            T.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
            T.RandomGrayscale(p=p_grayscale),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
    else:
        # Validation transform is identical for both experiments (apples-to-apples)
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
