import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_dataloader(batch_size=64, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # Download CIFAR-10 dataset if not available
    dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


if __name__ == '__main__':
    dl = get_dataloader()
    print(f"DataLoader created with {len(dl)} batches.")
