import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError

class MNISTDataset(BaseDataset):
    def __init__(self, data, targets):
        super().__init__()
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx].float() / 255.0  # 정규화
        
        # 데이터 형태 확인 및 변환
        if len(x.shape) == 1:  # (784,)
            x = x.view(1, 28, 28)  # (1, 28, 28)
        elif len(x.shape) == 2:  # (28, 28)
            x = x.unsqueeze(0)  # (1, 28, 28)
        
        y = self.targets[idx]
        return x, y

class RandomDataset(BaseDataset):
    def __init__(self, num_samples, num_features, num_classes):
        super().__init__()
        self.data = torch.randn(num_samples, num_features)
        self.targets = torch.randint(0, num_classes, (num_samples,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]