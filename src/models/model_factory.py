from typing import Union
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import timm

class ModelFactory:
    @staticmethod
    def create_model(cfg) -> Union[nn.Module, RandomForestClassifier, xgb.XGBClassifier]:
        model_type = cfg.model.type
        num_classes = cfg.data.num_classes

        if model_type == "mlp":
            layers = []
            prev_dim = cfg.data.num_features
            
            for dim in cfg.model.hidden_dims:
                layers.append(nn.Linear(prev_dim, dim))
                layers.append(nn.ReLU())
                prev_dim = dim
            
            layers.append(nn.Linear(prev_dim, cfg.data.num_classes))
            model = nn.Sequential(*layers)
            
        elif model_type == "efficientnet":
            model = timm.create_model(
                cfg.model.architecture,
                pretrained=cfg.model.pretrained,
                num_classes=cfg.data.num_classes,
                in_chans=cfg.model.in_channels
            )
            
        elif model_type == "vit":
            model = timm.create_model(
                cfg.model.architecture,
                pretrained=cfg.model.pretrained,
                num_classes=cfg.data.num_classes,
                in_chans=cfg.model.in_channels
            )
            
        elif model_type == "deit":
            model = timm.create_model(
                cfg.model.architecture,
                pretrained=cfg.model.pretrained,
                num_classes=cfg.data.num_classes,
                in_chans=cfg.model.in_channels
            )
            
        elif model_type == "random_forest":
            model = RandomForestClassifier(**cfg.model.model_params)
            return SklearnModelWrapper(model)
            
        elif model_type == "xgboost":
            xgb_params = {
                'objective': 'multi:softprob',
                'num_class': num_classes,
                **cfg.model.model_params
            }
            model = xgb.XGBClassifier(**xgb_params)
            return SklearnModelWrapper(model)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model 