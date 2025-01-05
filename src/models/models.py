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