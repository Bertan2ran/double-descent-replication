import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import matplotlib.pyplot as plt

seed = 41

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

transform = transforms.ToTensor()

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

print("Number of training images:", len(train_dataset))
print("Number of test images:", len(test_dataset))

image, label = train_dataset[0]

print("Shape of one image:", image.shape)
print("Label of first image:", label)

from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)

images, labels = next(iter(train_loader))

print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)

class FiveLayerCNN(nn.Module):
    def __init__(self, width):
        super().__init__()

        k = width

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, k, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(k),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1),

            # Block 2
            nn.Conv2d(k, 2 * k, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * k),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 3
            nn.Conv2d(2 * k, 4 * k, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * k),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 4
            nn.Conv2d(4 * k, 8 * k, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8 * k),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=8),
        )

        # Fifth layer: classifier
        self.classifier = nn.Linear(8 * k, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
model = FiveLayerCNN(width=8)

outputs = model(images)

print("Input batch shape:", images.shape)
print("Output shape:", outputs.shape)

num_parameters = sum(p.numel() for p in model.parameters())

print("Number of parameters:", num_parameters)
for k in [2, 4, 8, 16, 32, 64]:
    model = FiveLayerCNN(width=k)

    num_parameters = sum(p.numel() for p in model.parameters())

    print(f"Width {k}: {num_parameters:,} parameters")


def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    average_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total

    return average_loss, accuracy
def evaluate(model, data_loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    average_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total

    return average_loss, accuracy


def add_label_noise(dataset, noise_rate, seed=41):
    rng = np.random.default_rng(seed)

    targets = np.array(dataset.targets)

    num_samples = len(targets)
    num_noisy = int(noise_rate * num_samples)

    noisy_indices = rng.choice(
        num_samples,
        size=num_noisy,
        replace=False
    )

    for idx in noisy_indices:
        old_label = targets[idx]

        # choose one of the other 9 classes
        possible_labels = list(range(10))
        possible_labels.remove(old_label)

        new_label = rng.choice(possible_labels)
        targets[idx] = new_label

    dataset.targets = targets.tolist()

    return noisy_indices

noisy_indices = add_label_noise(
    train_dataset,
    noise_rate=0.20,
    seed=41
)

print("Number of noisy training labels:", len(noisy_indices))
widths = [2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 48, 64]

results = []

num_epochs = 250

for width in widths:

    print("\n" + "=" * 50)
    print(f"Training model with width k = {width}")
    print("=" * 50)

    model = FiveLayerCNN(width=width).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[100, 150],
        gamma=0.1
    )

    for epoch in range(num_epochs):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion
        )
        scheduler.step()

        # only print every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train acc: {train_acc:.2f}% | "
                f"Test acc: {test_acc:.2f}%"
            )

    num_parameters = sum(
        p.numel() for p in model.parameters()
    )

    train_error = 1 - train_acc / 100
    test_error = 1 - test_acc / 100

    results.append({
        "width": width,
        "parameters": num_parameters,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_error": train_error,
        "test_error": test_error
    })
print("\nFINAL RESULTS")

for result in results:
    print(
        f"Width {result['width']:>2} | "
        f"Parameters: {result['parameters']:>8,} | "
        f"Train acc: {result['train_accuracy']:.2f}% | "
        f"Test acc: {result['test_accuracy']:.2f}%"
    )

width_values = [r["width"] for r in results]
train_errors = [r["train_error"] for r in results]
test_errors = [r["test_error"] for r in results]

plt.figure(figsize=(8, 5))

plt.plot(
    width_values,
    train_errors,
    marker="o",
    label="Train error"
)

plt.plot(
    width_values,
    test_errors,
    marker="o",
    label="Test error"
)

plt.xlabel("CNN width k")
plt.ylabel("Error")
plt.title("Model-wise Double Descent on CIFAR-10")
plt.legend()
plt.grid()

plt.savefig("double_descent.png", dpi=300, bbox_inches="tight")
plt.close()