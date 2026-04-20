"""
Training Script for Meme Bully CNN Model

RUN THIS AFTER ADDING IMAGES TO data/ FOLDER

Usage:
    python train_cnn.py
"""

import os
import torch
from torch.utils.data import DataLoader, random_split
from cnn_model import MemeBullyCNN, MemeDataset, get_transforms, train_model
import argparse


def main():
    # Configuration
    data_dir = 'data'  # Your images should be in data/bully/ and data/non_bully/
    batch_size = 32
    num_epochs = 15
    learning_rate = 0.001
    train_split = 0.8  # 80% training, 20% validation

    print("="*60)
    print("🎭 Meme Bully CNN Training")
    print("="*60)

    # Check if data folder exists
    if not os.path.exists(data_dir):
        print(f"❌ Error: {data_dir}/ folder not found!")
        print("\nCreate this structure:")
        print("""
data/
├── bully/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── non_bully/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
        """)
        return

    # Load dataset
    print(f"\n📂 Loading images from {data_dir}/...")
    train_transform, val_transform = get_transforms()

    dataset = MemeDataset(data_dir, transform=train_transform)
    print(f"✅ Loaded {len(dataset)} images")

    if len(dataset) == 0:
        print("❌ No images found! Add images to data/bully/ and data/non_bully/")
        return

    # Split dataset
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"   - Training: {train_size} images")
    print(f"   - Validation: {val_size} images")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Create model
    print("\n🤖 Creating CNN Model (ResNet50 + Custom Head)...")
    model = MemeBullyCNN(num_classes=2, pretrained=True)
    print("✅ Model created")

    # Create models folder
    os.makedirs('models', exist_ok=True)

    # Train model
    print("\n🏋️  Starting training...")
    print("-"*60)

    trained_model, history = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )

    # Save final model
    torch.save(trained_model.state_dict(), 'models/final_model.pth')
    print("✅ Final model saved to models/final_model.pth")

    # Save training info
    with open('models/training_info.txt', 'w') as f:
        f.write("Meme Bully CNN Training Information\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total Images: {len(dataset)}\n")
        f.write(f"Training Images: {train_size}\n")
        f.write(f"Validation Images: {val_size}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Model: ResNet50 + Custom Head\n")
        f.write(f"Device: {device}\n")
        f.write(f"Best Validation Accuracy: {max(history['val_acc']):.2f}%\n")

    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"Model saved: models/best_model.pth")
    print(f"Final model saved: models/final_model.pth")
    print(f"Training info: models/training_info.txt")
    print("\nNext step: Run the Streamlit app!")
    print("  streamlit run app.py")
    print("="*60)


if __name__ == "__main__":
    main()
