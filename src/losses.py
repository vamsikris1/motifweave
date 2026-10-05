import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon) as proposed in:
    Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
    
    Pulls representations of the same source_id together while pushing
    representations of different source_ids apart.
    """
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Tensor of shape (B, embedding_dim), L2-normalized.
            labels: 1D Tensor of shape (B,) containing source_id labels.
        Returns:
            Scalar loss tensor.
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        # Ensure 1D labels
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError(f"Num labels {labels.shape[0]} does not match batch size {batch_size}")
            
        # Positive mask: 1 if labels match, 0 otherwise
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove self-contrast from positive mask
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask  # Positive pairs (excluding self)
        
        # Compute cosine similarity matrix scaled by temperature
        # Note: embeddings are L2 normalized, so dot product = cosine similarity
        similarity = torch.div(
            torch.matmul(embeddings, embeddings.T),
            self.temperature
        )
        
        # Numerical stability: subtract row-wise max
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()
        
        # Compute log-sum-exp over all negative and positive pairs (excluding self)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # Mean log-probability over positive pairs for each anchor
        mask_pos_pairs = mask.sum(1)
        valid_anchors = mask_pos_pairs > 0
        
        # Handle cases where an anchor has no positives in the batch
        mean_log_prob_pos = (mask * log_prob).sum(1)[valid_anchors] / (mask_pos_pairs[valid_anchors] + 1e-12)
        
        if valid_anchors.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=device)
            
        loss = -mean_log_prob_pos.mean()
        return loss
