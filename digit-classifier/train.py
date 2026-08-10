import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
from model import SimpleCNN

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 64
    epochs = 8
    learning_rate = 0.001

    # Data transformation: to tensor and normalize to roughly [-1, 1]
    # Standard MNIST normalization (mean=0.1307, std=0.3081)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load datasets
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    metrics = {
        "epochs": [],
        "train_loss": [],
        "test_loss": [],
        "test_accuracy": []
    }

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        epoch_test_loss = test_loss / len(test_loader.dataset)
        epoch_test_acc = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Test Loss: {epoch_test_loss:.4f} | Test Acc: {epoch_test_acc:.2f}%")
        
        metrics["epochs"].append(epoch + 1)
        metrics["train_loss"].append(epoch_train_loss)
        metrics["test_loss"].append(epoch_test_loss)
        metrics["test_accuracy"].append(epoch_test_acc)

    # Save model and metrics
    torch.save(model.state_dict(), "model.pth")
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Training complete. Model saved to model.pth, metrics saved to metrics.json")

if __name__ == "__main__":
    train()
