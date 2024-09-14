import torch

CONFIG = {
    # Device configuration
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    
    # Dataset configuration
    'dataset_root': 'data',
    'case_name': 'pglib_opf_case14_ieee',
    'split': 'train',
    
    # Model configuration
    'hidden_channels': 64,
    
    # Training configuration
    'batch_size': 32,
    'learning_rate': 0.01,
    'num_epochs': 10,
    
    # DCOPF solver configuration
    'dcopf_cost_factor': 10,
    
    # Paths
    'model_save_path': 'acopf_predictor_ieee14.pth'
}