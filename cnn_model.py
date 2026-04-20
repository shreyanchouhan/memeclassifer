"""
CNN Model for Meme Bully Classification
Uses Transfer Learning with ResNet50
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np


class MemeDataset(Dataset):
    """Custom dataset for loading meme images"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # Load bully images (label: 1)
        bully_dir = os.path.join(root_dir, 'bully')
        if os.path.exists(bully_dir):
            for img in os.listdir(bully_dir):
                if img.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    self.images.append(os.path.join(bully_dir, img))
                    self.labels.append(1)

        # Load non-bully images (label: 0)
        non_bully_dir = os.path.join(root_dir, 'non_bully')
        if os.path.exists(non_bully_dir):
            for img in os.listdir(non_bully_dir):
                if img.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    self.images.append(os.path.join(non_bully_dir, img))
                    self.labels.append(0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # If image can't be opened, return a black image
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        return image, label


class MemeBullyCNN(nn.Module):
    """
    CNN Model for Meme Bully Classification
    Uses ResNet50 as base with custom classification head
    """

    def __init__(self, num_classes=2, pretrained=True):
        super(MemeBullyCNN, self).__init__()

        # Load pre-trained ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)

        # Freeze early layers (transfer learning)
        for param in list(self.resnet.parameters())[:-2]:
            param.requires_grad = False

        # Replace classification head
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


def get_transforms():
    """Image preprocessing transformations"""

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """Train the CNN model"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_accuracy = 0
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        # Store history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['train_acc'].append(train_accuracy)
        training_history['val_acc'].append(val_accuracy)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f"✅ Best model saved! (Accuracy: {best_accuracy:.2f}%)")

        scheduler.step()

    print(f"\n{'='*50}")
    print(f"Training complete! Best Accuracy: {best_accuracy:.2f}%")
    print(f"Model saved to: models/best_model.pth")
    print(f"{'='*50}")

    return model, training_history


if __name__ == "__main__":
    print("CNN Model for Meme Bully Classification")
    print("This module is imported by train_cnn.py")
