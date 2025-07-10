import torch.nn.functional as F

def compute_cosine_similarity(query_feats, db_feats):
    query_norm = F.normalize(query_feats, dim=1)
    db_norm = F.normalize(db_feats, dim=1)
    return query_norm @ db_norm.T

def evaluate_retrieval(query_feats, db_feats, query_labels, db_labels, k=5):
    sim_matrix = compute_cosine_similarity(query_feats, db_feats)
    topk = sim_matrix.topk(k, dim=1).indices
    topk_labels = db_labels[topk]
    correct = (topk_labels == query_labels.unsqueeze(1)).float()
    return correct.sum(dim=1).mean().item()
