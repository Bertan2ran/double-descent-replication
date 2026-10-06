import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def inject_symmetric_noise(targets, noise_rate: float, num_classes: int = 10, seed: int = 42):
    if noise_rate <= 0.0:
        return targets
    
    rng = np.random.RandomState(seed)
    noisy_targets = list(targets)
    n = len(noisy_targets)
    mask = rng.rand(n) < noise_rate
    
    for idx in np.where(mask)[0]:
        orig = noisy_targets[idx]
        choices = [c for c in range(num_classes) if c != orig]
        noisy_targets[idx] = rng.choice(choices)
        
    return noisy_targets

def get_dataloaders(n_samples: int = 50000, noise_rate: float = 0.20, batch_size: int = 128, seed: int = 42):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    # deterministic subsampling of n samples
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(train_dataset))[:n_samples]
    sub_targets = [train_dataset.targets[i] for i in indices]

    # addding fixed label noise once at dataset instantiation
    noisy_sub_targets = inject_symmetric_noise(sub_targets, noise_rate=noise_rate, seed=seed)

    #  wrap the dataset with modified targets
    class NoisySubset(torch.utils.data.Dataset):
        def __init__(self, base_ds, indices, targets):
            self.base_ds = base_ds
            self.indices = indices
            self.targets = targets

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            img, _ = self.base_ds[self.indices[idx]]
            return img, self.targets[idx]

    train_subset = NoisySubset(train_dataset, indices, noisy_sub_targets)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader