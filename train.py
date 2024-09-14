import torch
import torch.nn.functional as F

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch.x_dict, batch.edge_index_dict)
        
        loss = 0
        loss_components = 0
        for node_type in out:
            if node_type in batch and hasattr(batch[node_type], 'y'):
                node_loss = F.mse_loss(out[node_type], batch[node_type].y)
                loss += node_loss
                loss_components += 1
        
        if loss_components > 0:
            loss = loss / loss_components  # Average loss across all node types
            
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)