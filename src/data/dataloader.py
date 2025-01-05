from torch.utils.data import DataLoader
from .datasets import DatasetFactory
import os
import psutil  # 시스템 정보 확인을 위해 추가

def create_dataloaders(cfg, train_data, val_data):
    """공통 DataLoader 생성 함수"""
    train_loader = DataLoader(
        train_data,
        batch_size=cfg.train.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=cfg.train.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers
    )
    
    print(f"Train Dataset initialized with {len(train_data)} samples ({cfg.data.train_val_split*100:.0f}%)")
    print(f"Val Dataset initialized with {len(val_data)} samples ({(1-cfg.data.train_val_split)*100:.0f}%)\n")
    
    return train_loader, val_loader

def get_optimal_num_workers():
    """시스템 환경에 따른 최적의 num_workers 반환"""
    # CPU 코어 수 확인
    cpu_count = os.cpu_count()
    # 사용 가능한 RAM (GB)
    available_memory = psutil.virtual_memory().available / (1024 ** 3)
    
    # 기본값으로 CPU 코어 수의 2배 설정
    optimal_workers = cpu_count * 2
    
    # 메모리가 16GB 이하인 경우 worker 수 제한
    if available_memory < 16:
        optimal_workers = min(optimal_workers, cpu_count)
    
    # 최소 1, 최대 CPU 코어 수의 4배로 제한
    optimal_workers = max(1, min(optimal_workers, cpu_count * 4))
    
    return int(optimal_workers)

def get_dataloaders(cfg):
    """데이터로더 팩토리 함수"""
    if cfg.data.name in ["mnist", "fashion_mnist", "cifar10", "svhn"]:
        # num_workers가 설정되지 않은 경우 최적값 계산
        if not cfg.data.get('num_workers'):
            cfg.data.num_workers = get_optimal_num_workers()
            print(f"Setting optimal num_workers to: {cfg.data.num_workers}")
            
        train_dataset, val_dataset = DatasetFactory.create_dataset(cfg)
        return create_dataloaders(cfg, train_dataset, val_dataset)
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.name}") 