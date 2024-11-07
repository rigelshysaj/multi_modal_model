# -*- coding: utf-8 -*-
"""Multimodal Regression Model"""

import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import open_clip
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from step1_pipeline import TabularPreprocessor
from utils import get_image_filename


# Set constants
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Multimodal preprocessor for Step 2
class MultimodalPreprocessor:
    """Preprocessor for multimodal data (Step 2)"""
    def __init__(self):
        self.tabular_preprocessor = TabularPreprocessor()
        self.image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def fit_transform(self, data):
        # Preprocess tabular data
        tabular_data = self.tabular_preprocessor.fit_transform(data)
        return tabular_data, data['description']

    def transform(self, data):
        tabular_data = self.tabular_preprocessor.transform(data)
        return tabular_data, data['description']

    def inverse_transform_target(self, y):
        return self.tabular_preprocessor.inverse_transform_target(y)

# Multimodal Dataset
class MultimodalDataset(Dataset):
    def __init__(self, tabular_data, descriptions, targets, image_folder, preprocess, tokenizer):
        self.tabular_data = tabular_data.reset_index(drop=True)
        self.descriptions = descriptions.reset_index(drop=True)
        if targets is not None:
            self.targets = targets.reset_index(drop=True)
        else:
            self.targets = None
        self.image_folder = image_folder
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def get_image_path(self, description):
        base_filename = get_image_filename(description)
        possible_extensions = ['.jpg', '.jpeg', '.png']
        for ext in possible_extensions:
            image_name = f"{base_filename}{ext}"
            image_path = os.path.join(self.image_folder, image_name)
            if os.path.isfile(image_path):
                return image_path
        return None

    def __getitem__(self, idx):
        # Tabular data
        tabular = torch.tensor(self.tabular_data.iloc[idx].values, dtype=torch.float32)

        # Text data
        text = self.descriptions.iloc[idx]

        # Image data
        image_path = self.get_image_path(text)
        if image_path:
            try:
                image = Image.open(image_path).convert('RGB')
                image = self.preprocess(image)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                image = torch.zeros_like(self.preprocess(Image.new('RGB', (224, 224))))
        else:
            # If image not found, use a placeholder
            image = torch.zeros_like(self.preprocess(Image.new('RGB', (224, 224))))

        if self.targets is not None:
            target = torch.tensor(self.targets.iloc[idx], dtype=torch.float32)
            return {
                'tabular': tabular,
                'text': text,
                'image': image,
                'target': target
            }
        else:
            return {
                'tabular': tabular,
                'text': text,
                'image': image
            }

    def __len__(self):
        return len(self.tabular_data)

# Multimodal Model for Step 2
class MultimodalModel(nn.Module):
    """Multimodal Model (Step 2)"""
    def __init__(self, tabular_dim, clip_model, preprocess, tokenizer):
        super().__init__()

        self.clip_model = clip_model
        self.preprocess = preprocess
        self.tokenizer = tokenizer

        # Tabular network
        self.tabular_network = nn.Sequential(
            nn.Linear(tabular_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Projection layers to map CLIP embeddings to a common dimension
        clip_embedding_dim = self.clip_model.text_projection.shape[1]
        self.text_projection = nn.Linear(clip_embedding_dim, 128)
        self.image_projection = nn.Linear(clip_embedding_dim, 128)

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, tabular, texts, images):
        # Tabular features
        tabular_features = self.tabular_network(tabular)

        # Text features
        text_tokens = self.tokenizer(texts).to(DEVICE)
        text_features = self.clip_model.encode_text(text_tokens)
        text_features = self.text_projection(text_features)

        # Image features
        image_features = self.clip_model.encode_image(images)
        image_features = self.image_projection(image_features)

        # Concatenate features
        combined_features = torch.cat([tabular_features, text_features, image_features], dim=1)
        output = self.fusion(combined_features)

        return output

# Step 2 Pipeline
class Step2Pipeline:
    """Complete pipeline for Step 2 (multimodal)"""
    def __init__(self):
        self.preprocessor = MultimodalPreprocessor()

    def create_model(self, tabular_dim, clip_model, preprocess, tokenizer):
        return MultimodalModel(tabular_dim, clip_model, preprocess, tokenizer).to(DEVICE)

    def train(self, data, images_folder):
        # Preprocess the data
        tabular_data, descriptions = self.preprocessor.fit_transform(data)
        targets = tabular_data['target'].reset_index(drop=True)

        # Drop 'target' from tabular_data
        tabular_data = tabular_data.drop('target', axis=1).reset_index(drop=True)

        # Convert tabular data to float32
        tabular_data = tabular_data.astype(np.float32)
        descriptions = descriptions.reset_index(drop=True)

        # Split indices for train/test
        train_indices, test_indices = train_test_split(
            np.arange(len(tabular_data)), test_size=0.2, random_state=RANDOM_SEED
        )

        # Prepare CLIP model and tokenizer
        model_name = 'ViT-B-32'
        pretrained = 'laion2b_s34b_b79k'
        clip_model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)

        # Create datasets
        train_dataset = MultimodalDataset(
            tabular_data.iloc[train_indices],
            descriptions.iloc[train_indices],
            targets.iloc[train_indices],
            images_folder,
            self.preprocessor.image_transforms,
            tokenizer
        )

        test_dataset = MultimodalDataset(
            tabular_data.iloc[test_indices],
            descriptions.iloc[test_indices],
            targets.iloc[test_indices],
            images_folder,
            self.preprocessor.image_transforms,
            tokenizer
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2
        )

        # Initialize model
        tabular_dim = len(tabular_data.columns)
        print(f"Tabular dimension: {tabular_dim}")
        model = self.create_model(tabular_dim, clip_model, preprocess, tokenizer)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Training loop
        best_val_loss = float('inf')
        for epoch in range(EPOCHS):
            # Training
            model.train()
            train_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()

                output = model(
                    batch['tabular'].to(DEVICE),
                    batch['text'],  # Pass raw text
                    batch['image'].to(DEVICE)
                )

                loss = criterion(output.squeeze(), batch['target'].to(DEVICE))
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            predictions = []
            actuals = []
            with torch.no_grad():
                for batch in test_loader:
                    output = model(
                        batch['tabular'].to(DEVICE),
                        batch['text'],
                        batch['image'].to(DEVICE)
                    )
                    loss = criterion(output.squeeze(), batch['target'].to(DEVICE))
                    val_loss += loss.item()

                    # Collect predictions and actual values
                    preds = output.squeeze().cpu().numpy()
                    targets = batch['target'].cpu().numpy()

                    # Apply inverse transformation
                    preds_inv = self.preprocessor.inverse_transform_target(preds)
                    targets_inv = self.preprocessor.inverse_transform_target(targets)

                    predictions.extend(preds_inv)
                    actuals.extend(targets_inv)

            train_loss /= len(train_loader)
            val_loss /= len(test_loader)

            # Calculate metrics
            mae = mean_absolute_error(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))

            print(f'Epoch {epoch+1}/{EPOCHS}:')
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'Validation MAE: {mae:.4f}')
            print(f'Validation RMSE: {rmse:.4f}')

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_multimodal_model.pth')
