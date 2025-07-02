import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.dino import DINOModel
from models.dino_loss import DINOLoss
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import torch.nn.utils as nn_utils

def save_models(student_model, teacher_model, epoch, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(student_model.state_dict(), os.path.join(save_dir, f"student_epoch_{epoch}.pth"))
    torch.save(teacher_model.state_dict(), os.path.join(save_dir, f"teacher_epoch_{epoch}.pth"))
    print(f"[Checkpoint] Saved student and teacher models at epoch {epoch}.")

def train_dino(dataloader, epochs=100, out_dim=256, save_dir="checkpoints", log_dir="logs"):
    import os
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    from models.dino import DINOModel
    from models.dino_loss import DINOLoss
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    import torch.nn.utils as nn_utils

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    model = DINOModel(out_dim=out_dim).to(device)
    criterion = DINOLoss(out_dim=out_dim).to(device)
    criterion.writer = writer  # Enable logging
    criterion.global_step = 0

    optimizer = torch.optim.AdamW(model.student.parameters(), lr=1e-4, weight_decay=0.05)

    warmup_epochs = 10
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for views in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            views = [v.to(device) for v in views]
            student_views = views
            teacher_views = views[:2]

            optimizer.zero_grad()

            student_outs = model(student_views)
            with torch.no_grad():
                teacher_outs = model(teacher_views, use_teacher=True)
            loss = criterion(student_outs, teacher_outs)

            loss.backward()

            # Log gradient norm
            total_grad_norm = 0.0
            for p in model.student.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            writer.add_scalar("Gradients/global_norm", total_grad_norm, criterion.global_step)

            nn_utils.clip_grad_norm_(model.student.parameters(), max_norm=3.0)
            optimizer.step()

            model.momentum_update()
            model.update_center(teacher_outs, epoch, epochs)

            # Log center norm
            center_norm = model.center.norm().item()
            writer.add_scalar("Center/norm", center_norm, criterion.global_step)

            total_loss += loss.item()
            criterion.global_step += 1

        avg_loss = total_loss / len(dataloader)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")

        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_models(model.student, model.teacher, epoch="best", save_dir=save_dir)

        if (epoch + 1) % 10 == 0:
            save_models(model.student, model.teacher, epoch=epoch + 1, save_dir=save_dir)

    writer.close()
    print("Training completed.")
