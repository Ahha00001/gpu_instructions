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
encoder.student.load_state_dict(torch.load("checkpoints/real/student_epoch_best.pth", map_location=device))
encoder.eval()
for param in encoder.parameters():
    param.requires_grad = True  # Unfreeze encoder for fine-tuning

# Extract features
train_feats, train_labels = extract_features(encoder, train_loader, device)
val_feats, val_labels = extract_features(encoder, val_loader, device)

# Debug prints
print("Train features shape:", train_feats.shape)
print("Train labels shape:", train_labels.shape)
print("Train label distribution:", torch.bincount(train_labels))
print("Val label distribution:", torch.bincount(val_labels))
print("Feature sample:", train_feats[0])

# Check for all-zero or constant features
print("Train features mean:", train_feats.mean().item(), "std:", train_feats.std().item())
print("Val features mean:", val_feats.mean().item(), "std:", val_feats.std().item())

# Logistic Regression sanity check
try:
    from sklearn.linear_model import LogisticRegression
    print("Training sklearn LogisticRegression on extracted features...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_feats.numpy(), train_labels.numpy())
    val_acc = clf.score(val_feats.numpy(), val_labels.numpy())
    print("Logistic Regression Val Acc:", val_acc)
except Exception as e:
    print("Logistic Regression sanity check failed:", e)

# Classification
feature_dim = train_feats.shape[1]
# Use a deeper/wider MLP for classification
mlp_classifier = train_classifier(
    train_feats, train_labels, val_feats, val_labels,
    num_classes=37, feature_dim=feature_dim,
    use_mlp=True, max_epochs=50, patience=10, save_dir="checkpoints/real_mlp_classifier",
)


