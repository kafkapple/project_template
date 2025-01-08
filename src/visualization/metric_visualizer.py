import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, 
    precision_recall_curve, 
    average_precision_score
)

class MetricVisualizer:
    """메트릭 시각화 전용 클래스"""
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_confusion_matrix(self, y_true, y_pred, phase='train', step=None):
        """Confusion Matrix 시각화"""
        cm = confusion_matrix(y_true, y_pred)
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        return wandb.Image(fig)
    
    def plot_pr_curve(self, y_true, y_prob, n_classes, phase='train'):
        """PR Curve 시각화"""
        fig = plt.figure(figsize=(10, 8))
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
        
        return wandb.Image(fig)

    def plot_learning_curves(self, metrics_history, current_epoch):
        """학습 커브 시각화"""
        # 기존 learning curve 플로팅 로직
        pass 