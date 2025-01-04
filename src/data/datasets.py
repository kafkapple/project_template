from torchvision import datasets, transforms
from torch.utils.data import random_split

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
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                  (0.2470, 0.2435, 0.2616))
            ])
            dataset = datasets.CIFAR10(cfg.data.data_dir, train=True,
                                     download=True, transform=transform)
            
            
        # 학습/검증 데이터 분할
        train_size = int(len(dataset) * cfg.data.train_val_split)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        return train_dataset, val_dataset