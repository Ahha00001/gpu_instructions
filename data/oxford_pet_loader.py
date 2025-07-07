import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_oxford_iiit_pet_loaders(data_dir="./data/oxford_pet", batch_size=64, num_workers=4, img_size=224):
    """
    Returns train and test DataLoaders for the Oxford-IIIT Pet dataset.
    """
    # Define transforms (resize to 224x224, normalize to ImageNet stats)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Download and create datasets
    train_dataset = datasets.OxfordIIITPet(
        root=data_dir,
        split="trainval",
        target_types="category",
        download=True,
        transform=transform,
    )
    test_dataset = datasets.OxfordIIITPet(
        root=data_dir,
        split="test",
        target_types="category",
        download=True,
        transform=transform,
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
