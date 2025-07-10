import torch
from preprocess_data import get_pet_dataloaders
from features import extract_features
from eval_classify import train_classifier
from eval_retrieval import evaluate_retrieval

# Replace this with your pretrained frozen ViT encoder
from models.dino import DINOModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Load dataset
train_loader, val_loader = get_pet_dataloaders()

# Step 2: Load encoder
encoder = DINOModel(out_dim=256).to(device)
encoder.load_state_dict(torch.load("../checkpoints/real/student_epoch_best.pth", map_location=device))
encoder.eval()
for param in encoder.parameters():
    param.requires_grad = False

# Step 3: Extract features
train_feats, train_labels = extract_features(encoder.student, train_loader, device)
val_feats, val_labels = extract_features(encoder.student, val_loader, device)

# Step 4: Classification
classifier = train_classifier(train_feats, train_labels, val_feats, val_labels, num_classes=37)

# Step 5: Retrieval
retrieval_score = evaluate_retrieval(val_feats, train_feats, val_labels, train_labels, k=5)

