import torch.nn.functional as F

def compute_cosine_similarity(query, db):
    query = F.normalize(query, dim=1)
    db = F.normalize(db, dim=1)
    return query @ db.T

def evaluate_retrieval(query_feats, db_feats, query_labels, db_labels, k=5):
    sim_matrix = compute_cosine_similarity(query_feats, db_feats)
    topk = sim_matrix.topk(k, dim=1).indices
    topk_labels = db_labels[topk]
    correct = (topk_labels == query_labels.unsqueeze(1)).float()
    return correct.sum(dim=1).mean().item()
