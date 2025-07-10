import torch
from preprocess_data import get_pet_dataloaders
from features import extract_features
from eval_classify import train_classifier
from eval_retrieval import evaluate_retrieval

from models.dino import DINOModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
train_loader, val_loader = get_pet_dataloaders()

# Load encoder
encoder = DINOModel(out_dim=256).to(device)
encoder.load_state_dict(torch.load("../checkpoints/real/student_epoch_best.pth", map_location=device))
encoder.eval()
for param in encoder.parameters():
    param.requires_grad = False

# Extract features
train_feats, train_labels = extract_features(encoder.student, train_loader, device)
val_feats, val_labels = extract_features(encoder.student, val_loader, device)

# Classification
linear_classifier = train_classifier(train_feats, train_labels, val_feats, val_labels, num_classes=37)
mlp_classifier = train_classifier(train_feats, train_labels, val_feats, val_labels, num_classes=37, use_mlp=True, save_dir="checkpoints/real_mlp_classifier")


