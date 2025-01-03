from torch.utils.data import DataLoader
from .datasets import DatasetFactory

def create_dataloaders(cfg, train_data, val_data):
    """공통 DataLoader 생성 함수"""
    train_loader = DataLoader(
        train_data,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers
    )
    
    print(f"Train Dataset initialized with {len(train_data)} samples ({cfg.data.train_val_split*100:.0f}%)")
    print(f"Val Dataset initialized with {len(val_data)} samples ({(1-cfg.data.train_val_split)*100:.0f}%)\n")
    
    return train_loader, val_loader

def get_dataloaders(cfg):
    """데이터로더 팩토리 함수"""
    if cfg.data.name in ["mnist", "fashion_mnist", "cifar10", "svhn"]:
        train_dataset, val_dataset = DatasetFactory.create_dataset(cfg)
        return create_dataloaders(cfg, train_dataset, val_dataset)
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.name}") 