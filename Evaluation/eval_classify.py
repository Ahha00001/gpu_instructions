import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay        

class LinearClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

def train_classifier(train_features, train_labels, val_features, val_labels, num_classes, save_dir="checkpoints/real_classifier"):
    os.makedirs(save_dir, exist_ok=True)
    model = LinearClassifier(train_features.shape[1], num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float('inf')
    best_epoch = -1

    for epoch in range(100):
        model.train()
        out = model(train_features)
        loss = criterion(out, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        val_out = model(val_features)
        val_loss = criterion(val_out, val_labels)
        _, preds = torch.max(val_out, 1)
        acc = (preds == val_labels).float().mean()

        # Save model if validation loss improves
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
    
    return model
