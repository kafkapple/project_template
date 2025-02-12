from torchvision import datasets, transforms
from torch.utils.data import random_split, Subset
import numpy as np

class DatasetFactory:
    @staticmethod
    def create_dataset(cfg):
        if cfg.data.name == "mnist":
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])
            dataset = datasets.MNIST(cfg.data.data_dir, train=True, 
                                  download=True, transform=transform)
            
        elif cfg.data.name == "fashion_mnist":
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])
            dataset = datasets.FashionMNIST(cfg.data.data_dir, train=True,
                                          download=True, transform=transform)
            
        elif cfg.data.name == "cifar10":
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), # RGB 각 채널별 평균
                                  (0.2470, 0.2435, 0.2616)) # RGB 각 채널별 표준편차
            ])
            dataset = datasets.CIFAR10(cfg.data.data_dir, train=True,
                                     download=True, transform=transform)
 
        # Debug 모드인 경우 데이터 샘플링
        if cfg.logger.wandb.job_type == "debug":
            total_size = len(dataset)
            sample_size = int(total_size * cfg.debug.data_ratio)
            indices = np.random.choice(total_size, sample_size, replace=False)
            dataset = Subset(dataset, indices)
            print(f"\nDebug mode: Using {sample_size} samples ({cfg.debug.data_ratio*100:.1f}% of data)")
        
        # 학습/검증 데이터 분할
        train_size = int(len(dataset) * cfg.data.train_val_split)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        return train_dataset, val_dataset

