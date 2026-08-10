import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
from model import SimpleCNN

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    misclassified = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Find misclassified examples
            mask = predicted != labels
            if mask.any():
                wrong_imgs = images[mask].cpu()
                wrong_preds = predicted[mask].cpu()
                actual_labels = labels[mask].cpu()
                
                for i in range(len(wrong_imgs)):
                    if len(misclassified) < 8:
                        misclassified.append({
                            "image": wrong_imgs[i].squeeze().numpy(),
                            "predicted": wrong_preds[i].item(),
                            "actual": actual_labels[i].item()
                        })

    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='gray_r', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Saved confusion_matrix.png")
    
    # Plot Misclassified Examples
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle('Misclassified Examples', fontsize=16)
    for i, ax in enumerate(axes.flat):
        if i < len(misclassified):
            item = misclassified[i]
            # De-normalize image for viewing
            img = item["image"] * 0.3081 + 0.1307
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Pred: {item['predicted']} | Act: {item['actual']}")
            ax.axis('off')
    plt.tight_layout()
    plt.savefig('misclassified_examples.png')
    print("Saved misclassified_examples.png")

if __name__ == "__main__":
    evaluate()
