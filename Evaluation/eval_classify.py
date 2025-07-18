import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, f1_score

from torch.utils.tensorboard import SummaryWriter

class LinearClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.model = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.model(x)

class FlexibleMLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.3):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dims[-1], num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def top_k_accuracy(output, labels, k=1):
    _, pred = output.topk(k, dim=1)
    correct = pred.eq(labels.view(-1, 1).expand_as(pred))
    return correct.any(dim=1).float().mean().item()

def train_classifier(train_features, train_labels, val_features, val_labels,
                     num_classes, feature_dim, use_mlp=False, max_epochs=30, patience=5, save_dir="checkpoints/real_classifier"):
    
    os.makedirs(save_dir, exist_ok=True)
   
    model = FlexibleMLPClassifier(feature_dim, [512, 256], num_classes) if use_mlp else LinearClassifier(feature_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0
    best_model = None
    wait = 0
    train_acc_list, val_acc_list = [], []
    writer = SummaryWriter(log_dir="logs")

    for epoch in range(max_epochs):
        model.train()
        output = model(train_features)
        loss = criterion(output, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Train Acc
        train_preds = torch.argmax(output, dim=1)
        train_acc = (train_preds == train_labels).float().mean()

        model.eval()
        val_output = model(val_features)
        val_preds = torch.argmax(val_output, dim=1)
        val_acc = (val_preds == val_labels).float().mean()

        # Top-1 and Top-5 accuracy
        val_top1 = top_k_accuracy(val_output, val_labels, k=1)
        val_top5 = top_k_accuracy(val_output, val_labels, k=5)

        # Recall and F1-score (macro)
        recall = recall_score(val_labels.cpu(), val_preds.cpu(), average='macro', zero_division=0)
        fscore = f1_score(val_labels.cpu(), val_preds.cpu(), average='macro', zero_division=0)

        train_acc_list.append(train_acc.item())
        val_acc_list.append(val_acc.item()) 

        #Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
            wait = 0                    
        else:
            wait += 1
            if wait >= patience:
                break
        
        writer.add_scalar("Train/Accuracy", train_acc.item(), epoch)
        writer.add_scalar("Validation/Accuracy", val_acc.item(), epoch)
        writer.add_scalar("Validation/Top1", val_top1, epoch)
        writer.add_scalar("Validation/Top5", val_top5, epoch)
        writer.add_scalar("Validation/Recall", recall, epoch)
        writer.add_scalar("Validation/F1", fscore, epoch)

        if save_dir:
            torch.save(model.state_dict(), save_dir + f"/epoch_{epoch+1}_val_acc_{val_acc:.4f}.pth")  


    # Plot
    plt.plot(train_acc_list, label="Train Acc")
    plt.plot(val_acc_list, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_dir.replace(".pt", "_acc.png")) if save_dir else plt.show()

    return model

#Confusion Matrix
def plot_confusion_matrix(model, features, labels, class_names=None):
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(features), dim=1)
    cm = confusion_matrix(labels.cpu(), preds.cpu())
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=90)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()