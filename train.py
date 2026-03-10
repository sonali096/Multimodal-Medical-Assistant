"""
Training module for Multimodal Medical Assistant
Handles model training with proper logging and checkpointing
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
from datetime import datetime

# Import model components
from models import MultimodalFusionModel
from utils import calculate_metrics, print_evaluation_report


def setup_logging():
    """
    Setup logging configuration for training.
    Creates log directory if it doesn't exist.
    """
    os.makedirs('./logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'./logs/training_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Dictionary containing configuration
    """
    if not os.path.exists(config_path):
        print(f"⚠️  Config file not found: {config_path}")
        print("📝 Using default configuration...")
        return get_default_config()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_default_config():
    """
    Returns default configuration if config file is not found.
    
    Returns:
        Dictionary with default settings
    """
    return {
        'training': {
            'num_epochs': 10,
            'batch_size': 16,
            'learning_rate': 1e-4,
            'device': 'cpu'
        },
        'model': {
            'num_classes': 5,
            'fusion_type': 'concatenation',
            'hidden_dim': 512,
            'dropout': 0.3
        }
    }


def get_device(config):
    """
    Get training device (CUDA, MPS, or CPU).
    
    Args:
        config: Configuration dictionary
        
    Returns:
        torch.device object
    """
    device_name = config['training'].get('device', 'cpu')
    
    if device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using CUDA: {torch.cuda.get_device_name(0)}")
    elif device_name == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ Using MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("✅ Using CPU")
    
    return device


def train_model(config_path):
    """
    Main training function for the multimodal model.
    
    Args:
        config_path: Path to configuration file
    """
    # Setup
    logger = setup_logging()
    config = load_config(config_path)
    device = get_device(config)
    
    logger.info("Starting training process...")
    logger.info(f"Configuration: {config}")
    
    # Create directories for outputs
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    # Initialize model
    model_config = config.get('model', get_default_config()['model'])
    model = MultimodalFusionModel(
        num_classes=model_config.get('num_classes', 5),
        fusion_type=model_config.get('fusion_type', 'concatenation'),
        hidden_dim=model_config.get('hidden_dim', 512),
        dropout=model_config.get('dropout', 0.3)
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized with {num_params:,} parameters")
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training'].get('learning_rate', 1e-4)
    )
    
    # Training loop placeholder
    num_epochs = config['training'].get('num_epochs', 10)
    
    print("\n" + "="*70)
    print("📚 TRAINING SETUP COMPLETE")
    print("="*70)
    print(f"Model Parameters: {num_params:,}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning Rate: {config['training'].get('learning_rate', 1e-4)}")
    print("="*70)
    print()
    
    logger.warning("⚠️  Note: Actual training requires dataset loading.")
    logger.info("To train with real data:")
    logger.info("1. Prepare your medical dataset (text + images + labels)")
    logger.info("2. Implement data loading in utils/data_loader.py")
    logger.info("3. Update this training script with your data")
    
    # Save model architecture info
    model_info = {
        'num_parameters': num_params,
        'config': model_config,
        'training_config': config['training']
    }
    
    import json
    with open('./results/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logger.info("✅ Training setup completed successfully")
    logger.info("Model information saved to: ./results/model_info.json")
    
    return model


if __name__ == "__main__":
    """
    Direct execution for training.
    """
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    train_model(config_file)
