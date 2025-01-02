import torch
import torch.nn as nn
import timm
import numpy as np

class BaseModel(nn.Module):
    """모든 모델의 기본 클래스"""
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.flatten = False  # 기본값: 이미지 형태 유지
    
    def preprocess(self, x):
        """입력 데이터 전처리"""
        if self.flatten and len(x.shape) > 2:
            # (B, C, H, W) -> (B, C*H*W)
            return x.view(x.size(0), -1)
        return x
    
    def forward(self, x):
        x = self.preprocess(x)
        return self.model(x)

class TimmModel(BaseModel):
    """timm 기반 모델 래퍼 클래스"""
    def __init__(self, model_name, num_classes, pretrained=True, **kwargs):
        super().__init__(num_classes)
        kwargs.pop('in_chans', None)  # in_chans가 있다면 제거
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=1,  # MNIST는 항상 1채널
            **kwargs
        )
    
    def preprocess(self, x):
        """Vision 모델을 위한 전처리"""
        # 입력 형태 확인 및 변환
        if len(x.shape) == 3:  # (B, H, W)
            x = x.unsqueeze(1)  # (B, 1, H, W)
        elif len(x.shape) == 2:  # (B, HW)
            x = x.view(-1, 1, 28, 28)  # (B, 1, H, W)
        
        # 이미지가 (B, C, H, W) 형태인지 확인
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input tensor, got shape {x.shape}")
        
        # 224x224로 리사이즈
        x = torch.nn.functional.interpolate(
            x,
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )
        return x

class EfficientNet(TimmModel):
    """EfficientNet 모델"""
    def __init__(self, num_classes, variant='b0', **kwargs):
        super().__init__(
            f'efficientnet_{variant}',
            num_classes=num_classes,
            **kwargs
        )

class ViT(TimmModel):
    """Vision Transformer 모델"""
    def __init__(self, num_classes, variant='tiny_patch16_224', **kwargs):
        super().__init__(
            f'vit_{variant}',
            num_classes=num_classes,
            **kwargs
        )

class DeiT(TimmModel):
    """Data-efficient Image Transformer"""
    def __init__(self, num_classes, variant='tiny_patch16_224', **kwargs):
        super().__init__(
            f'deit_{variant}',
            num_classes=num_classes,
            **kwargs
        )

class SimpleClassifier(BaseModel):
    """MLP 모델"""
    def __init__(self, input_dim, num_classes, hidden_dims, dropout):
        super().__init__(num_classes)
        self.flatten = True  # MLP는 flatten된 입력 필요
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)

class SklearnModelWrapper:
    """scikit-learn/xgboost 모델을 PyTorch와 비슷한 인터페이스로 감싸는 래퍼 클래스"""
    def __init__(self, model):
        self.model = model
        self.device = 'cpu'  # sklearn/xgboost 모델은 CPU만 사용
        self.is_fitted = False
    
    def to(self, device):
        print(f"Warning: {type(self.model).__name__} only supports CPU")
        return self
    
    def train(self):
        pass  # sklearn 모델은 train 모드가 없음
    
    def eval(self):
        pass  # sklearn 모델은 eval 모드가 없음
    
    def fit(self, X, y):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        
        if not self.is_fitted:
            # XGBoost의 경우 classes를 직접 설정할 수 없으므로 
            # 모델 초기화 시 설정되어 있어야 함
            self.model.fit(X, y)
            self.is_fitted = True
    
    def predict_proba(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        return self.model.predict_proba(X)
    
    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        proba = self.predict_proba(x)
        return torch.from_numpy(proba).float().to(x.device) 