from typing import Union
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from .models import (
    SimpleClassifier, SklearnModelWrapper,
    EfficientNet, ViT, DeiT
)

class ModelFactory:
    @staticmethod
    def create_model(cfg) -> Union[nn.Module, RandomForestClassifier, xgb.XGBClassifier]:
        model_type = cfg.model.type
        model_params = cfg.model.model_params
        num_features = cfg.data.num_features
        num_classes = cfg.data.num_classes

        if model_type == "mlp":
            return SimpleClassifier(
                input_dim=num_features,
                num_classes=num_classes,
                **model_params
            )
        elif model_type == "efficientnet":
            return EfficientNet(
                num_classes=num_classes,
                **model_params
            )
        elif model_type == "vit":
            return ViT(
                num_classes=num_classes,
                **model_params
            )
        elif model_type == "deit":
            return DeiT(
                num_classes=num_classes,
                **model_params
            )
        elif model_type == "random_forest":
            model = RandomForestClassifier(**model_params)
            return SklearnModelWrapper(model)
        elif model_type == "xgboost":
            xgb_params = {
                'objective': 'multi:softprob',
                'num_class': num_classes,
                **model_params
            }
            model = xgb.XGBClassifier(**xgb_params)
            return SklearnModelWrapper(model)
        else:
            raise ValueError(f"Unknown model type: {model_type}") 