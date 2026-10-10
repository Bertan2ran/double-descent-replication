import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse
import math
import csv
import random

# Import the architecture from the models directory
from models.cnn import FiveLayerCNN

def set_seed(seed=42):
    """Ensures deterministic subsampling and noise injection."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def inject_label_noise(targets, noise_prob, num_classes=10):
    targets = np.array(targets)
    num_samples = len(targets)
    num_noisy = int(noise_prob * num_samples)
    
    noisy_indices = np.random.choice(num_samples, num_noisy, replace=False)
    for idx in noisy_indices:
        current_label = targets[idx]
        possible_labels = [i for i in range(num_classes) if i != current_label]
        targets[idx] = np.random.choice(possible_labels)
        
    return targets.tolist()

def main():
    parser = argparse.ArgumentParser(description="Double Descent Replication")
    parser.add_argument('--width', type=int, default=128, help="Multiplier k for CNN width")
    parser.add_argument('--samples', type=int, default=50000, help="Number of training samples")
    parser.add_argument('--noise', type=float, default=0.2, help="Label noise probability")
    parser.add_argument('--wd', type=float, default=0.0, help="Weight decay")
    parser.add_argument('--epochs', type=int, default=4000, help="Number of epochs to train")
    parser.add_argument('--out_csv', type=str, default="results.csv", help="Path to output CSV")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Deterministic Subsampling
    if args.samples < 50000:
        trainset.data = trainset.data[:args.samples]
        trainset.targets = trainset.targets[:args.samples]

    # Symmetric Label Noise
    if args.noise > 0:
        trainset.targets = inject_label_noise(trainset.targets, noise_prob=args.noise)

    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    model = FiveLayerCNN(k=args.width).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.0, weight_decay=args.wd)
    
    def inverse_sqrt_lr(step):
        return 1.0 / math.sqrt(1 + (step // 512))
    
    scheduler = LambdaLR(optimizer, lr_lambda=inverse_sqrt_lr)
    criterion = nn.CrossEntropyLoss()

    with open(args.out_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'train_loss', 'train_err', 'test_loss', 'test_err'])

        for epoch in range(args.epochs):
            model.train()
            train_loss_sum, correct_train, total_train = 0, 0, 0
            
            for data, target in train_loader:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss_sum += loss.item() * data.size(0)
                _, predicted = output.max(1)
                total_train += target.size(0)
                correct_train += predicted.eq(target).sum().item()
                
            train_err = 1.0 - (correct_train / total_train)
            avg_train_loss = train_loss_sum / total_train

            model.eval()
            test_loss_sum, correct_test, total_test = 0, 0, 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    test_loss_sum += loss.item() * data.size(0)
                    _, predicted = output.max(1)
                    total_test += target.size(0)
                    correct_test += predicted.eq(target).sum().item()
                    
            test_err = 1.0 - (correct_test / total_test)
            avg_test_loss = test_loss_sum / total_test

            writer.writerow([epoch, avg_train_loss, train_err, avg_test_loss, test_err])
            file.flush()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Train Err: {train_err:.4f} | Test Err: {test_err:.4f}")

if __name__ == "__main__":
    main()
