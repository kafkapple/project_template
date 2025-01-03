from typing import Dict, List, Tuple
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
    def __init__(self, cfg, metric_names: List[str]):
        self.cfg = cfg
        self.metric_names = metric_names
        self.learning_curves = {}  # 학습 곡선 데이터 저장용
    
    def _to_numpy(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    def _get_sample_predictions(self, images, y_true, y_pred, n_samples=1) -> List[Tuple]:
        """각 클래스별로 정답/오답 샘플 수집"""
        samples = []
        n_classes = len(np.unique(y_true))
        
        for class_idx in range(n_classes):
            # 정답 샘플
            correct_idx = np.where((y_true == class_idx) & (y_pred == class_idx))[0]
            if len(correct_idx) > 0:
                correct_samples = np.random.choice(
                    correct_idx, 
                    size=min(n_samples, len(correct_idx)), 
                    replace=False
                )
                for idx in correct_samples:
                    samples.append((
                        images[idx],
                        int(y_true[idx]),
                        int(y_pred[idx]),
                        "Correct"
                    ))
            
            # 오답 샘플
            wrong_idx = np.where((y_true == class_idx) & (y_pred != class_idx))[0]
            if len(wrong_idx) > 0:
                wrong_samples = np.random.choice(
                    wrong_idx,
                    size=min(n_samples, len(wrong_idx)),
                    replace=False
                )
                for idx in wrong_samples:
                    samples.append((
                        images[idx],
                        int(y_true[idx]),
                        int(y_pred[idx]),
                        "Wrong"
                    ))
        
        return samples

    def calculate(self, outputs, labels, phase='train', step=None, logger=None, images=None, loss=None) -> Dict[str, float]:
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
        if logger is not None:
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
            
            # 학습 곡선 데이터 저장 (train과 val 모두)
            for metric_name, value in metrics.items():
                if metric_name not in self.learning_curves:
                    self.learning_curves[metric_name] = {'train': [], 'val': []}
                if phase in ['train', 'val']:
                    self.learning_curves[metric_name][phase].append((step, value))
            
            # loss도 learning curves에 추가
            if loss is not None:  # loss가 전달된 경우에만 추가
                if 'loss' not in self.learning_curves:
                    self.learning_curves['loss'] = {'train': [], 'val': []}
                self.learning_curves['loss'][phase].append((step, loss))
            
            # 매 단계(train/val)마다 learning curves 업데이트
            self._plot_learning_curves(logger, step)
            
            # 예측 샘플 테이블 생성 (이미지가 제공된 경우)
            if images is not None and phase == 'val':
                samples = self._get_sample_predictions(
                    self._to_numpy(images.permute(0, 2, 3, 1)),  # [B, C, H, W] -> [B, H, W, C]로 변경
                    y_true,
                    y_pred
                )
                
                table_data = []
                for img, true_label, pred_label, status in samples:
                    table_data.append([
                        wandb.Image(img),  # 이미 [H, W, C] 형태로 변환된 이미지
                        true_label,
                        pred_label,
                        status
                    ])
                
                logger.log({
                    f"{phase}/prediction_samples": wandb.Table(
                        data=table_data,
                        columns=["Image", "True Label", "Predicted Label", "Status"]
                    )
                }, step=step)
        
        return metrics

    def _plot_learning_curves(self, logger, step):
        """학습 곡선 그리기"""
        learning_curve_metrics = self.cfg.train.metrics.learning_curve
        n_plots = len(learning_curve_metrics)
        
        # 1행 2열 레이아웃으로 변경
        fig, axes = plt.subplots(1, n_plots, figsize=(15, 5))
        if n_plots == 1:
            axes = [axes]
        
        for ax, metric_name in zip(axes, learning_curve_metrics):
            if metric_name in self.learning_curves:
                # Train 데이터 플롯
                train_data = self.learning_curves[metric_name]['train']
                if train_data:
                    epochs, values = zip(*train_data)
                    ax.plot(epochs, values, label=f'Train {metric_name}', marker='o')
                
                # Validation 데이터 플롯
                val_data = self.learning_curves[metric_name]['val']
                if val_data:
                    epochs, values = zip(*val_data)
                    ax.plot(epochs, values, label=f'Val {metric_name}', marker='o')
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric_name)
                ax.set_title(f'{metric_name} Learning Curve')
                ax.legend()
                ax.grid(True)
        
        plt.tight_layout()
        logger.log({
            "learning_curves": wandb.Image(plt)
        }, step=step)
        plt.close()

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