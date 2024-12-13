import torch
from TrafficSignsDataset import TrafficSignsDataset
from utils import load_data
from torch.utils.data import DataLoader
from utils import plot_confusion_matrix

classes =['Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100', 'Speed Limit 110', 'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30', 'Speed Limit 40', 'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70', 'Speed Limit 80', 'Speed Limit 90', 'Stop']

def evaluate_model_on_test_data(model, image_dir, label_dir, criterion, device, transform):

    test_data, test_labels = load_data(image_dir, label_dir)
    test_dataset = TrafficSignsDataset(test_data, test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_loss /= len(test_loader)
    test_accuracy = correct / total
    plot_confusion_matrix(model, test_loader, classes, device)

    return test_loss, test_accuracy