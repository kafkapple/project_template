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
import matplotlib.gridspec as gridspec
from pathlib import Path

class MetricCalculator:
    def __init__(self, cfg, metric_names: List[str]):
        self.cfg = cfg
        self.metric_names = metric_names
        self.learning_curves = {}  # epoch 단위 메트릭
        self.step_history = {     # step 단위 메트릭
            'loss': [],
            'learning_rate': []
        }
        # 메트릭 저장 경로 설정
        self.metrics_dir = Path(cfg.dirs.outputs) / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.total_epochs = cfg.train.training.epochs  # 전체 에폭 수 저장
    
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
        # LightGBM의 경우 outputs가 (n_samples,) 형태일 수 있음
        if len(outputs.shape) == 1:
            predicted = outputs
            y_prob = outputs  # 이미 확률값인 경우
        else:
            _, predicted = outputs.max(1)
            y_prob = self._to_numpy(outputs)
        
        y_true = self._to_numpy(labels)
        y_pred = self._to_numpy(predicted)
        
        # 기본 메트릭 계산
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # 필요한 메트릭 선택
        metrics = {}
        # sklearn의 키 이름과 매핑
        metric_mapping = {
            "accuracy": "accuracy",
            "f1": "f1-score",
            "f1": "f1-score",
            "precision": "precision",
            "recall": "recall"
        }
        
        for metric_name in self.metric_names:
            if metric_name == "loss":  # loss는 별도 처리
                metrics[metric_name] = loss if loss is not None else 0.0
            elif metric_name == "accuracy":
                metrics[metric_name] = report["accuracy"]
            else:
                # macro avg에서 올바른 키 이름으로 매핑하여 가져오기
                mapped_name = metric_mapping.get(metric_name, metric_name)
                metrics[metric_name] = report["macro avg"][mapped_name]
        
        # Classification Report - verbose_metrics가 true일 때 콘솔에 출력
        if self.cfg.train.verbose_metrics:
            class_report = classification_report(y_true, y_pred, output_dict=False)
            print(f"\n{'='*20} {phase.upper()} Classification Report {'='*20}")
            print(class_report)
            print('='*80)
        
        # epoch 단위 메트릭 저장 - phase 처리 수정
        if phase in ['train', 'val']:
            for metric_name in self.cfg.train.metrics.learning_curve.metrics:
                if metric_name not in self.learning_curves:
                    self.learning_curves[metric_name] = {'train': [], 'val': []}
                
                # phase prefix 제거하고 메트릭 값 가져오기
                metric_key = metric_name
                if metric_name in metrics:
                    value = metrics[metric_key]
                elif f"{phase}/{metric_name}" in metrics:  # phase prefix가 있는 경우
                    value = metrics[f"{phase}/{metric_name}"]
                else:
                    value = 0.0
                    
                self.learning_curves[metric_name][phase].append((step, value))
            
            # loss도 learning curves에 추가
            if 'loss' not in self.learning_curves:
                self.learning_curves['loss'] = {'train': [], 'val': []}
            self.learning_curves['loss'][phase].append((step, loss if loss is not None else 0.0))
            
            # learning rate는 train phase에서만 추가
            if phase == 'train' and 'learning_rate' in metrics:
                if 'learning_rate' not in self.learning_curves:
                    self.learning_curves['learning_rate'] = {'train': []}
                self.learning_curves['learning_rate']['train'].append((step, metrics['learning_rate']))
        
        # wandb 로깅이 필요한 경우에만 추가 시각화
        if logger is not None:
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            fig = plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            logger.log({f"{phase}/confusion_matrix": wandb.Image(plt)}, step=step)
            self._save_figure(fig, f'confusion_matrix_{phase}.png', step)
            plt.close()
            
            # PR Curve for each class
            n_classes = outputs.shape[1]
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
            logger.log({f"{phase}/pr_curve": wandb.Image(plt)}, step=step)
            self._save_figure(fig, f'pr_curve_{phase}.png', step)
            plt.close()
            
            # Classification Report as Table
            class_report = classification_report(y_true, y_pred, output_dict=False)
            logger.log({f"{phase}/classification_report": wandb.Table(
                columns=["Metric"],
                data=[[class_report]]
            )}, step=step)
            
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
            
            # learning curves 업데이트
            self._plot_learning_curves(logger, step)
        
        return metrics

    def _plot_learning_curves(self, logger, step):
        """모든 메트릭을 한 행에 나란히 표시하는 학습 곡선"""
        metrics_to_plot = self.cfg.train.metrics.learning_curve.metrics
        n_metrics = len(metrics_to_plot)
        
        # 모든 subplot을 한 행에 배치
        fig = plt.figure(figsize=(6 * (n_metrics), 5))
        gs = gridspec.GridSpec(1, n_metrics, figure=fig)
        
        # 각 메트릭별 subplot 생성
        for idx, metric_name in enumerate(metrics_to_plot):
            ax = fig.add_subplot(gs[0, idx])
            
            if metric_name in self.learning_curves:
                # Train 데이터
                train_data = self.learning_curves[metric_name]['train']
                if train_data:
                    epochs, values = zip(*train_data)
                    ax.plot(epochs, values, 
                           'b-', 
                           label=f'Train',
                           marker='o',
                           alpha=0.7)
                
                # Validation 데이터
                val_data = self.learning_curves[metric_name]['val']
                if val_data:
                    epochs, values = zip(*val_data)
                    ax.plot(epochs, values, 
                           'r--', 
                           label=f'Val',
                           marker='s',
                           alpha=0.7)
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric_name)
                ax.set_title(f'{metric_name}')
                ax.legend()
                ax.grid(True)
                
                # Loss인 경우 y축 범위 조정
                if metric_name == 'loss':
                    ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        # wandb 로깅
        logger.log({
            "learning_curves": wandb.Image(plt)
        }, step=step)
        
        # 마지막 에폭에서만 로컬에 저장
        self._save_figure(fig, 'learning_curves.png', step)
        plt.close()

    def add_step_metrics(self, metrics: dict, step: int):
        """step 단위 메트릭을 히스토리에 추가"""
        for metric_name in self.cfg.train.metrics.step.metrics:
            if metric_name in metrics:
                self.step_history[metric_name].append(
                    (step, metrics[metric_name])
                )

    def log_step_history(self, logger):
        """학습 종료 후 step 히스토리 로깅"""
        if not self.step_history['loss']:
            return
        
        # Loss와 Learning Rate를 하나의 그래프로 통합
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()
        
        # Loss 플롯 (왼쪽 y축)
        steps, values = zip(*self.step_history['loss'])
        ax1.plot(steps, values, 'b-', label='Loss', alpha=0.7)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Learning Rate 플롯 (오른쪽 y축)
        if self.step_history['learning_rate']:
            steps, values = zip(*self.step_history['learning_rate'])
            ax2.plot(steps, values, 'g-', label='Learning Rate', alpha=0.7)
            ax2.set_ylabel('Learning Rate', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            ax2.set_yscale('log')
        
        # 범례 통합
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.title('Training Progress')
        plt.grid(True)
        plt.tight_layout()
        
        # wandb 로깅
        logger.log({
            "step_history": wandb.Image(plt)
        })
        
        # 학습 종료 시 로컬에 저장
        self._save_figure(fig, 'step_history.png')
        plt.close()

    def _save_figure(self, fig, filename: str, step=None):
        """마지막 에폭에서만 figure 저장"""
        if step is not None and step != self.total_epochs:
            return
        save_path = self.metrics_dir / filename
        fig.savefig(save_path)
        print(f"Figure saved to {save_path}")

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