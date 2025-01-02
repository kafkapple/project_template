from typing import Dict, List
import torch
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_curve, average_precision_score
)
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

class MetricCalculator:
    def __init__(self, metric_names: List[str]):
        self.metric_names = metric_names
    
    def _to_numpy(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    def calculate(self, outputs, labels, phase='train', step=None, logger=None) -> Dict[str, float]:
        _, predicted = outputs.max(1)
        y_true = self._to_numpy(labels)
        y_pred = self._to_numpy(predicted)
        y_prob = self._to_numpy(outputs)
        
        # 기본 메트릭 계산
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # 필요한 메트릭 선택
        metrics = {}
        # sklearn의 키 이름과 매핑
        metric_mapping = {
            "accuracy": "accuracy",
            "f1": "f1-score",        # 'f1' -> 'f1-score'로 매핑
            "precision": "precision",
            "recall": "recall"
        }
        
        for metric_name in self.metric_names:
            if metric_name == "accuracy":
                metrics[metric_name] = report["accuracy"]
            else:
                # macro avg에서 올바른 키 이름으로 매핑하여 가져오기
                mapped_name = metric_mapping.get(metric_name, metric_name)
                metrics[metric_name] = report["macro avg"][mapped_name]
        
        # wandb 로깅이 필요한 경우에만 추가 시각화
        if logger is not None and phase == 'val':
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            logger.log({f"{phase}/confusion_matrix": wandb.Image(plt)}, step=step)
            plt.close()
            
            # PR Curve for each class
            n_classes = outputs.shape[1]
            plt.figure(figsize=(10, 8))
            for i in range(n_classes):
                precision, recall, _ = precision_recall_curve(
                    (y_true == i).astype(int),
                    y_prob[:, i]
                )
                ap = average_precision_score((y_true == i).astype(int), y_prob[:, i])
                plt.plot(recall, precision, label=f'Class {i} (AP={ap:.2f})')
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            logger.log({f"{phase}/pr_curve": wandb.Image(plt)}, step=step)
            plt.close()
            
            # Classification Report as Table
            class_report = classification_report(y_true, y_pred, output_dict=False)
            logger.log({f"{phase}/classification_report": wandb.Table(
                columns=["Metric"],
                data=[[class_report]]
            )}, step=step)
        
        return metrics

class ModelCheckpointer:
    def __init__(self, cfg, save_dir):
        self.best_metric = cfg.train.best_model.metric
        self.mode = cfg.train.best_model.mode
        self.save_dir = save_dir
        self.best_value = float('-inf') if self.mode == 'max' else float('inf')
        
        if '/' in self.best_metric:
            self.phase, self.metric_name = self.best_metric.split('/')
        else:
            self.phase = None
            self.metric_name = self.best_metric
            
    def is_better(self, metrics: Dict[str, float]) -> bool:
        current_value = metrics[self.metric_name]
        if self.mode == 'max':
            is_better = current_value > self.best_value
        else:
            is_better = current_value < self.best_value
            
        if is_better:
            self.best_value = current_value
            
        return is_better 