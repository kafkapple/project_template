from typing import Union
import torch.nn as nn
import timm
import torch

class ModelFactory:
    @staticmethod
    def create_model(cfg) -> nn.Module:
        model_type = cfg.model.type
        num_classes = cfg.data.num_classes

        if model_type in ["vit", "deit"]:
            try:
                model = timm.create_model(
                    cfg.model.architecture,
                    pretrained=cfg.model.pretrained,
                    num_classes=num_classes,
                    in_chans=cfg.model.in_channels
                )
                print(f"Successfully loaded {model_type.upper()} model: {cfg.model.architecture}")
                return model
                
            except Exception as e:
                print(f"Error loading {model_type.upper()} model: {str(e)}")
                raise
                
        elif model_type == "mlp":
            layers = []
            layers.append(nn.Flatten())
            
            input_dim = 224 * 224 * 3
            prev_dim = input_dim
            
            for dim in cfg.model.hidden_dims:
                layers.append(nn.Linear(prev_dim, dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(cfg.model.dropout))
                prev_dim = dim
            
            layers.append(nn.Linear(prev_dim, num_classes))
            model = nn.Sequential(*layers)
            
        elif model_type == "efficientnet":
            model = timm.create_model(
                cfg.model.architecture,
                pretrained=cfg.model.pretrained,
                num_classes=num_classes,
                in_chans=cfg.model.in_channels
            )
            
        elif model_type == "vit":
            model = timm.create_model(
                cfg.model.architecture,
                pretrained=cfg.model.pretrained,
                num_classes=num_classes,
                in_chans=cfg.model.in_channels
            )
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model 