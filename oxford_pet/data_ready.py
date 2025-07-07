
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import random
from utils.transforms_presets import generate_multicrop_views
import torch

class AFHQDataWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]["image"]
        return generate_multicrop_views(img)
        
def create_dataloaders(batch_size=128, num_workers=4, seed=42):
    synthetic_ds = load_dataset("bitmind/AFHQ___stable-diffusion-xl-base-1.0", split="train")
    real_ds = load_dataset("huggan/AFHQ", split="train")

    # Use torch Generator for reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)

    synthetic_indices = random.sample(range(len(synthetic_ds)), 15000)
    real_indices = random.sample(range(len(real_ds)), 15000)

    synthetic = synthetic_ds.select(synthetic_indices)
    real = real_ds.select(real_indices)

    synthetic_data = AFHQDataWrapper(synthetic)
    real_data = AFHQDataWrapper(real)

    synthetic_loader = DataLoader(synthetic_data, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=True)
    real_loader = DataLoader(real_data, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=True)

    return synthetic_loader, real_loader