import torch

def extract_features(encoder, dataloader, device):
    encoder.eval()
    encoder.to(device)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            feats = encoder(imgs)
            all_features.append(feats.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_features), torch.cat(all_labels)
