import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from train import train
from TrafficSignClassifier import TrafficSignClassifier
from TrafficSignsDataset import TrafficSignsDataset
from evaluate import evaluate_model_on_test_data
from utils import load_data, plot_losses

def main():
    num_classes = 15
    batch_size = 64
    epochs = 20
    learning_rate = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_image_dir = "Dataset/output/train/images"
    train_label_dir = "Dataset/output/train/labels"
    train_data, train_labels = load_data(train_image_dir, train_label_dir)
    print(f"Loaded {len(train_data)} training images with labels.")

    valid_image_dir = "Dataset/output/valid/images"
    valid_label_dir = "Dataset/output/valid/labels"
    valid_data, valid_labels = load_data(valid_image_dir, valid_label_dir)
    print(f"Loaded {len(valid_data)} validation images with labels.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = TrafficSignsDataset(train_data, train_labels, transform=transform)
    valid_dataset = TrafficSignsDataset(valid_data, valid_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = TrafficSignClassifier(num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = train(model, train_loader, valid_loader, criterion, optimizer, device, epochs=epochs)

    torch.save(model.state_dict(), "traffic_sign_classifier.pth")

    test_image_dir = "Dataset/output/test/images"
    test_label_dir = "Dataset/output/test/labels"
    testSet_loss, testSet_accuracy = evaluate_model_on_test_data(
        model, test_image_dir, test_label_dir, criterion, device, transform
    )
    print(f"Test Loss: {testSet_loss:.4f}, Test Accuracy: {testSet_accuracy:.4f}")

    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()