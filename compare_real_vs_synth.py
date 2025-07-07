import torch
import torch.nn as nn
from models.dino import DINOModel
from oxford_pet_loader import get_oxford_iiit_pet_loaders
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score

# Helper: Load DINO weights

def load_dino_weights(model, checkpoint_path):
    state = torch.load(checkpoint_path, map_location='cpu')
    if 'student' in state:
        model.student.load_state_dict(state['student'])
    if 'student_projector' in state:
        model.student_projector.load_state_dict(state['student_projector'])
    return model

# Helper: Feature extraction

def extract_features(model, loader, device):
    features, labels = [], []
    with torch.no_grad():
        for images, lbls in tqdm(loader, desc="Extracting features"):
            images = images.to(device)
            feats = model.student_projector(model.student(images))
            features.append(feats.cpu())
            labels.append(lbls)
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)

# Helper: Linear classifier training

def train_linear_classifier(dino, train_loader, val_loader, num_classes, device, epochs=10):
    classifier = nn.Linear(256, num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        classifier.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                feats = dino.student_projector(dino.student(images))
            logits = classifier(feats)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += images.size(0)
        print(f"Train Loss: {total_loss/total_samples:.4f}, Acc: {total_correct/total_samples:.4f}")
        # Validation
        classifier.eval()
        val_loss, val_correct, val_samples = 0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                feats = dino.student_projector(dino.student(images))
                logits = classifier(feats)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_samples += images.size(0)
        print(f"Val Loss: {val_loss/val_samples:.4f}, Acc: {val_correct/val_samples:.4f}")
    return classifier

# Helper: Retrieval mAP

def compute_map(query_feats, query_labels, db_feats, db_labels):
    query_feats = query_feats / query_feats.norm(dim=1, keepdim=True)
    db_feats = db_feats / db_feats.norm(dim=1, keepdim=True)
    sims = torch.mm(query_feats, db_feats.t())
    mAPs = []
    for i in range(query_feats.size(0)):
        # Exclude self-match if query and db are the same set
        if torch.equal(query_feats, db_feats):
            sims_i = sims[i].clone()
            sims_i[i] = -float('inf')
        else:
            sims_i = sims[i]
        sorted_idx = torch.argsort(sims_i, descending=True)
        true_labels = (db_labels[sorted_idx] == query_labels[i]).numpy().astype(int)
        ap = average_precision_score(true_labels, np.arange(len(true_labels), 0, -1))
        mAPs.append(ap)
    return np.mean(mAPs)

# Main comparison

def evaluate_dino(checkpoint_path, name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_oxford_iiit_pet_loaders(
        data_dir="./oxford_pet/images", batch_size=64, num_workers=4, img_size=224, val_split=0.2, seed=42
    )
    dino = DINOModel(backbone_name="vit_small_patch16_224", out_dim=256)
    dino = load_dino_weights(dino, checkpoint_path)
    dino.eval()
    for p in dino.student.parameters():
        p.requires_grad = False
    for p in dino.student_projector.parameters():
        p.requires_grad = False
    dino = dino.to(device)
    # Linear classifier
    num_classes = len(train_loader.dataset.dataset.class_to_idx)
    classifier = train_linear_classifier(dino, train_loader, val_loader, num_classes, device, epochs=10)
    # Top-1 accuracy on val
    classifier.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            feats = dino.student_projector(dino.student(images))
            logits = classifier(feats)
            val_correct += (logits.argmax(1) == labels).sum().item()
            val_total += images.size(0)
    top1_acc = val_correct / val_total
    print(f"{name} - Top-1 Accuracy: {top1_acc:.4f}")
    # Retrieval mAP
    val_feats, val_labels = extract_features(dino, val_loader, device)
    train_feats, train_labels = extract_features(dino, train_loader, device)
    map_val_to_train = compute_map(val_feats, val_labels, train_feats, train_labels)
    print(f"{name} - Retrieval mAP (val->train): {map_val_to_train:.4f}")
    return top1_acc, map_val_to_train

if __name__ == "__main__":
    # Paths to checkpoints
    real_ckpt = "./checkpoints/real/student_epoch_best.pth"
    synth_ckpt = "./checkpoints/synthetic/student_epoch_best.pth"
    print("Evaluating real-pretrained model...")
    real_acc, real_map = evaluate_dino(real_ckpt, "Real Pretraining")
    print("Evaluating synthetic-pretrained model...")
    synth_acc, synth_map = evaluate_dino(synth_ckpt, "Synthetic Pretraining")
    print("\nSummary:")
    print(f"Real Pretraining:    Top-1 Acc = {real_acc:.4f}, mAP = {real_map:.4f}")
    print(f"Synthetic Pretraining: Top-1 Acc = {synth_acc:.4f}, mAP = {synth_map:.4f}")
