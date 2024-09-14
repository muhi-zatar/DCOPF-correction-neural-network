from config import CONFIG
import torch
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader
from models.acopf_predictor import ACOPFPredictor
from utils.data_preparation import prepare_data
from train import train

print(f"Using device: {CONFIG['device']}")

if __name__ == "__main__":
    # Load and process dataset
    train_ds = OPFDataset(root=CONFIG['dataset_root'], case_name=CONFIG['case_name'], split=CONFIG['split'])
    processed_train_ds = prepare_data(train_ds, CONFIG['device'])
    
    # Create DataLoader
    train_loader = DataLoader(processed_train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    
    # Initialize model
    sample_data = processed_train_ds[0].to(CONFIG['device'])
    model = ACOPFPredictor(hidden_channels=CONFIG['hidden_channels'], metadata=sample_data.metadata()).to(CONFIG['device'])
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Training loop
    for epoch in range(CONFIG['num_epochs']):
        loss = train(model, train_loader, optimizer, CONFIG['device'])
        print(f'Epoch {epoch+1:03d}, Loss: {loss:.4f}')
    
    # Save the model
    torch.save(model.state_dict(), CONFIG['model_save_path'])
    
    # Demonstration of prediction
    model.eval()
    with torch.no_grad():
        test_data = processed_train_ds[0].to(CONFIG['device'])
        prediction = model(test_data.x_dict, test_data.edge_index_dict)
        print("\nPredicted ACOPF solution for generators:")
        print(prediction['generator'].cpu().numpy())
        print("\nActual ACOPF solution for generators:")
        print(test_data['generator'].y.cpu().numpy())