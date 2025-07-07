
from oxford_pet.data_ready import create_dataloaders
from train.dino_train import train_dino

synthetic_loader, real_loader = create_dataloaders(batch_size=24, num_workers=4)


train_dino(
    dataloader=synthetic_loader,
    epochs=100,
    out_dim=256,
    save_dir="checkpoints/synthetic",
    log_dir="logs/synthetic"
)


# Jupyter Cell 4
train_dino(
    dataloader=real_loader,
    epochs=100,
    out_dim=256,
    save_dir="checkpoints/real",
    log_dir="logs/real"
)
