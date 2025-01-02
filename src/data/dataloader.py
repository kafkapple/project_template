import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path
from .datasets import MNISTDataset, RandomDataset

def get_mnist_dataloaders(cfg):
    # 데이터셋 디렉토리 설정
    dataset_root = Path(cfg.dirs.dataset)
    dataset_dir = dataset_root / cfg.data.name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # MNIST 데이터셋 다운로드
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(root=str(dataset_dir), train=True, 
                         download=True, transform=transform)
    
    # 학습/검증 데이터 분할
    train_size = int(len(mnist) * cfg.data.train_val_split)
    val_size = len(mnist) - train_size
    train_dataset, val_dataset = random_split(
        mnist, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.project.seed)
    )
    
    # 커스텀 데이터셋으로 변환
    train_data = MNISTDataset(
        train_dataset.dataset.data[train_dataset.indices],
        train_dataset.dataset.targets[train_dataset.indices]
    )
    val_data = MNISTDataset(
        val_dataset.dataset.data[val_dataset.indices],
        val_dataset.dataset.targets[val_dataset.indices]
    )
    
    return create_dataloaders(cfg, train_data, val_data)

def get_random_dataloaders(cfg):
    # 데이터 크기 계산
    total_size = cfg.data.total_size
    train_size = int(total_size * cfg.data.train_val_split)
    val_size = total_size - train_size
    
    # 데이터셋 생성
    train_data = RandomDataset(
        num_samples=train_size,
        num_features=cfg.data.num_features,
        num_classes=cfg.data.num_classes
    )
    val_data = RandomDataset(
        num_samples=val_size,
        num_features=cfg.data.num_features,
        num_classes=cfg.data.num_classes
    )
    
    return create_dataloaders(cfg, train_data, val_data)

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
    if cfg.data.name == "mnist":
        return get_mnist_dataloaders(cfg)
    elif cfg.data.name == "random_dataset":
        return get_random_dataloaders(cfg)
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.name}") 