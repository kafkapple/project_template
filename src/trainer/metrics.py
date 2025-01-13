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
    _instances = {}
    _shared_learning_curves = None
    _shared_step_history = None  # step 단위 메트릭을 공유하기 위한 클래스 변수
    
    def __new__(cls, cfg, metric_names: List[str], train_steps_per_epoch: int):
        key = id(cfg)
        if key not in cls._instances:
            cls._instances[key] = super(MetricCalculator, cls).__new__(cls)
            # 공유 데이터 초기화 (첫 인스턴스에서만)
            if cls._shared_learning_curves is None:
                cls._shared_learning_curves = {
                    metric_name: {
                        'train': [],
                        'val': []
                    }
                    for metric_name in cfg.train.metrics.learning_curve.metrics
                }
                
                # step_history 초기화 - 기본 메트릭
                cls._shared_step_history = {
                    'train_loss': [],      # train phase loss
                    'val_loss': [],        # validation phase loss
                    'learning_rate': []    # 공유되는 learning rate
                }
                
                # step.metrics에 정의된 추가 메트릭들도 phase별로 초기화
                for metric_name in cfg.train.metrics.step.metrics:
                    if metric_name != 'loss' and metric_name != 'learning_rate':
                        cls._shared_step_history[f'train_{metric_name}'] = []
                        cls._shared_step_history[f'val_{metric_name}'] = []
                
                print(f"[DEBUG] Initialized shared data structures")
                print(f"[DEBUG] Step history keys: {list(cls._shared_step_history.keys())}")
                cls._steps_per_epoch = train_steps_per_epoch
        return cls._instances[key]

    def __init__(self, cfg, metric_names: List[str], train_steps_per_epoch: int):
        if hasattr(self, 'initialized'):
            return
            
        self.cfg = cfg
        self.metric_names = metric_names
        self.steps_per_epoch = train_steps_per_epoch
        self.learning_curves = self._shared_learning_curves
        self.step_history = self._shared_step_history
        
        self.metrics_dir = Path(cfg.dirs.outputs) / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.total_epochs = cfg.train.training.epochs
        
        self.initialized = True
    
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

    def calculate(self, outputs, labels, phase='train', step=None, logger=None, images=None, loss=None, learning_rate=None) -> Dict[str, float]:
        # LightGBM의 경우 outputs가 (n_samples,) 형태일 수 있음
        if len(outputs.shape) == 1:
            predicted = outputs
            y_prob = outputs  # 이미 확률값인 경우
        else:
            _, predicted = outputs.max(1)
            y_prob = self._to_numpy(outputs)
        
        y_true = self._to_numpy(labels)
        y_pred = self._to_numpy(predicted)
        
        # sklearn의 키 이름과 매핑
        metric_mapping = {
            "accuracy": "accuracy",
            "f1": "f1-score",
            "precision": "precision",
            "recall": "recall"
        }
        
        # 기본 메트릭 계산
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        base_metrics = {}
        
        for metric_name in self.metric_names:
            if metric_name == "loss":
                base_metrics[metric_name] = loss if loss is not None else 0.0
            elif metric_name == "learning_rate":
                base_metrics[metric_name] = learning_rate if learning_rate is not None else 0.0
            elif metric_name == "accuracy":
                base_metrics[metric_name] = report["accuracy"]
            else:
                mapped_name = metric_mapping.get(metric_name, metric_name)
                base_metrics[metric_name] = report["macro avg"][mapped_name]
        
        # Debug print 추가
        print(f"\n[DEBUG] Phase: {phase}")
        print(f"[DEBUG] Base metrics: {base_metrics}")
        print(f"[DEBUG] Learning curve metrics to track: {self.cfg.train.metrics.learning_curve.metrics}")
        
        # epoch 단위 메트릭 저장
        if phase in ['train', 'val']:
            for metric_name in self.cfg.train.metrics.learning_curve.metrics:
                value = None
                if metric_name == 'loss':
                    value = loss
                elif metric_name == 'learning_rate' and phase == 'train':  # learning rate는 train phase에서만 기록
                    value = learning_rate
                else:
                    value = base_metrics.get(metric_name)
                
                if value is not None:
                    print(f"[DEBUG] Adding {phase} metric: {metric_name}={value} at step {step}")
                    # 이미 __init__에서 초기화된 딕셔너리에 추가
                    self.learning_curves[metric_name][phase].append((step, value))
        
        # step 단위 메트릭 저장 - train phase에서만
        if phase == 'train' and self.cfg.train.metrics.step.enabled:
            if loss is not None:
                self.step_history['train_loss'].append((step, loss))
            if learning_rate is not None:
                self.step_history['learning_rate'].append((step, learning_rate))
            
            # 다른 메트릭들은 phase별로 구분하여 저장
            for metric_name in self.cfg.train.metrics.step.metrics:
                if metric_name in base_metrics:
                    metric_key = f'{phase}_{metric_name}'
                    if metric_key not in self.step_history:
                        self.step_history[metric_key] = []
                    self.step_history[metric_key].append(
                        (step, base_metrics[metric_name])
                    )
        
        # validation phase의 loss 저장
        elif phase == 'val' and loss is not None:
            self.step_history['val_loss'].append((step, loss))
        
        # Classification Report - verbose_metrics가 true일 때 콘솔에 출력
        if self.cfg.train.verbose_metrics:
            class_report = classification_report(y_true, y_pred, output_dict=False)
            print(f"\n{'='*20} {phase.upper()} Classification Report {'='*20}")
            print(class_report)
            print('='*80)
        
        # wandb 로깅이 필요한 경우에만 추가 시각화
        if logger is not None:
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            fig = plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # PyTorch Lightning WandbLogger는 log_metrics 사용
            logger.log_metrics({
                f"{phase}/confusion_matrix": wandb.Image(plt)
            }, step=step)
            self._save_figure(fig, f'confusion_matrix_{phase}.png', step)
            plt.close()
            
            # PR Curve
            fig = plt.figure(figsize=(10, 8))
            for i in range(outputs.shape[1]):
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
            
            logger.log_metrics({
                f"{phase}/pr_curve": wandb.Image(plt)
            }, step=step)
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
            
            # 각 phase가 끝날 때마다 learning curves 업데이트
            self._plot_learning_curves(logger, step)
        
        return base_metrics

    def _plot_learning_curves(self, logger, step):
        """에포크 단위 학습 곡선"""
        metrics_to_plot = self.cfg.train.metrics.learning_curve.metrics
        
        # Debug print 추가
        print("\n[DEBUG] Plotting learning curves at step:", step)
        print("[DEBUG] Memory address of learning_curves:", id(self.learning_curves))
        print("[DEBUG] Full learning curves data:")
        for metric in metrics_to_plot:
            if metric in self.learning_curves:
                print(f"\n{metric}:")
                print(f"  Train data: {self.learning_curves[metric]['train']}")
                print(f"  Val data: {self.learning_curves[metric]['val']}")
        
        fig = plt.figure(figsize=(6 * len(metrics_to_plot), 5))
        gs = gridspec.GridSpec(1, len(metrics_to_plot), figure=fig)
        
        for idx, metric_name in enumerate(metrics_to_plot):
            ax = fig.add_subplot(gs[0, idx])
            
            if metric_name in self.learning_curves:
                # Train 데이터
                train_data = self.learning_curves[metric_name]['train']
                if train_data:
                    epochs, values = zip(*train_data)
                    ax.plot(epochs, values, 'b-', label='Train', marker='o', alpha=0.7)
                
                # Validation 데이터
                val_data = self.learning_curves[metric_name]['val']
                if val_data:
                    epochs, values = zip(*val_data)
                    ax.plot(epochs, values, 'r--', label='Val', marker='s', alpha=0.7)
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric_name)
                ax.set_title(f'{metric_name}')
                ax.legend()
                ax.grid(True)
                
                if metric_name == 'loss':
                    ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        # wandb 로깅 - epoch/ 아래에 저장
        logger.log({
            "epoch/learning_curves": wandb.Image(plt)
        }, step=step)
        
        # 마지막 에폭에서만 로컬에 저장
        self._save_figure(fig, 'epoch_learning_curves.png', step)
        plt.close()

    def add_step_metrics(self, metrics: dict, step: int, phase='train'):
        """step 단위 메트릭을 히스토리에 추가 - phase 구분"""
        # 이미 해당 step에 대한 데이터가 있는지 확인
        def is_step_recorded(key, step):
            if key in self.step_history and self.step_history[key]:
                recorded_steps = [x[0] for x in self.step_history[key]]
                return step in recorded_steps
            return False

        print(f"\n[DEBUG] Adding step metrics - Phase: {phase}, Step: {step}")
        
        if 'loss' in metrics:
            loss_key = f'{phase}_loss'
            loss_value = metrics['loss']
            
            # validation의 경우 이전 step의 데이터 삭제 (최신 데이터만 유지)
            if phase == 'val':
                # 현재 에포크의 시작 step
                epoch_start_step = (step // self.steps_per_epoch) * self.steps_per_epoch
                # 이전 에포크의 데이터 유지
                self.step_history[loss_key] = [
                    (s, v) for s, v in self.step_history[loss_key] 
                    if s < epoch_start_step
                ]
            
            # 중복 기록 방지
            if not is_step_recorded(loss_key, step):
                self.step_history[loss_key].append((step, loss_value))
                print(f"[DEBUG] Added new {loss_key}: ({step}, {loss_value:.4f})")
            else:
                print(f"[DEBUG] Skipped duplicate {loss_key} for step {step}")
        
        # learning rate는 공유 (train phase에서만 추가)
        if phase == 'train' and 'learning_rate' in metrics:
            lr_value = metrics['learning_rate']
            
            # 중복 기록 방지
            if not is_step_recorded('learning_rate', step):
                self.step_history['learning_rate'].append((step, lr_value))
                print(f"[DEBUG] Added new learning_rate: ({step}, {lr_value:.6f})")
            else:
                print(f"[DEBUG] Skipped duplicate learning_rate for step {step}")
        
        # 주기적으로 데이터 일관성 체크
        if step % 100 == 0:  # 100 스텝마다
            self._check_data_consistency()

    def _check_data_consistency(self):
        """데이터 일관성 체크를 위한 헬퍼 메서드"""
        print("\n[DEBUG] Data Consistency Check:")
        
        # step history 데이터 체크
        for key in ['train_loss', 'val_loss', 'learning_rate']:
            if key in self.step_history:
                data = self.step_history[key]
                if data:
                    steps, values = zip(*data)
                    print(f"\n{key}:")
                    print(f"  Total points: {len(data)}")
                    print(f"  Step range: {min(steps)} to {max(steps)}")
                    print(f"  Value range: {min(values):.6f} to {max(values):.6f}")
                    print(f"  Last 3 points: {data[-3:] if len(data) >= 3 else data}")
                else:
                    print(f"\n{key}: Empty")

    def log_step_history(self, logger):
        """스텝 단위 히스토리 로깅 - train/val 구분"""
        if not (self.step_history['train_loss'] or self.step_history['val_loss']):
            print("[DEBUG] No loss data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
        
        def apply_smoothing(data, window_size):
            """데이터 스무딩 헬퍼 함수"""
            # 데이터 정렬
            sorted_data = sorted(data, key=lambda x: x[0])
            steps, values = zip(*sorted_data)
            steps, values = np.array(steps), np.array(values)
            
            if len(values) > window_size:
                # 중심 이동 평균 계산
                weights = np.ones(window_size) / window_size
                # 'valid' 모드 대신 'same' 모드 사용하여 원본과 같은 길이 유지
                smooth_values = np.convolve(values, weights, mode='same')
                
                # 경계 처리 개선
                half_window = window_size // 2
                # 시작 부분
                for i in range(half_window):
                    window = values[:(i + half_window + 1)]
                    smooth_values[i] = np.mean(window)
                # 끝 부분
                for i in range(len(values) - half_window, len(values)):
                    window = values[(i - half_window):]
                    smooth_values[i] = np.mean(window)
                
                return steps, values, smooth_values
            return steps, values, values
        
        # 위쪽 그래프: Loss (train/val 구분)
        if self.step_history['train_loss']:
            # Train loss 스무딩 (배치 단위라 큰 윈도우 사용)
            steps, raw_values, smooth_values = apply_smoothing(
                self.step_history['train_loss'], 
                window_size=50
            )
            print(f"[DEBUG] Train data shapes - steps: {steps.shape}, raw: {raw_values.shape}, smooth: {smooth_values.shape}")
            
            # 원본 데이터는 매우 투명하게
            ax1.plot(steps, raw_values, 'b-', alpha=0.1, label='Train Loss (raw)')
            # 스무딩된 데이터
            ax1.plot(steps, smooth_values, 'b-', label='Train Loss (smoothed)', alpha=0.8, linewidth=2)
        
        if self.step_history['val_loss']:
            # Validation loss 스무딩 (에포크 단위라 작은 윈도우 사용)
            steps, raw_values, smooth_values = apply_smoothing(
                self.step_history['val_loss'], 
                window_size=3  # validation은 더 작은 윈도우 사용
            )
            print(f"[DEBUG] Val data shapes - steps: {steps.shape}, raw: {raw_values.shape}, smooth: {smooth_values.shape}")
            print(f"[DEBUG] Val loss data points: {list(zip(steps, raw_values))}")
            
            # 원본 데이터
            ax1.plot(steps, raw_values, 'r--', alpha=0.2, label='Val Loss (raw)', marker='o', markersize=4)
            # 스무딩된 데이터
            ax1.plot(steps, smooth_values, 'r-', label='Val Loss (smoothed)', alpha=0.8, linewidth=2)
        
        # y축 범위 설정 (이상치 제외)
        all_values = []
        if self.step_history['train_loss']:
            _, values = zip(*self.step_history['train_loss'])
            all_values.extend(values)
        if self.step_history['val_loss']:
            _, values = zip(*self.step_history['val_loss'])
            all_values.extend(values)
            
        if all_values:
            values = np.array(all_values)
            # 상하위 1% 제외한 범위 사용
            vmin, vmax = np.percentile(values, [1, 99])
            margin = (vmax - vmin) * 0.1  # 10% 여유 공간
            ax1.set_ylim(max(0, vmin - margin), vmax + margin)
        
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 아래쪽 그래프: Learning Rate (변경 없음)
        if self.step_history['learning_rate']:
            lr_data = sorted(self.step_history['learning_rate'], key=lambda x: x[0])
            steps, values = zip(*lr_data)
            ax2.plot(steps, values, 'g-', label='Learning Rate', alpha=0.8, linewidth=2)
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Learning Rate')
            ax2.set_yscale('log')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # wandb 로깅
        logger.log({
            "step/training_history": wandb.Image(plt)
        })
        
        self._save_figure(fig, 'step_training_history.png')
        plt.close()

        # 플롯 후 데이터 요약
        print("\n[DEBUG] Plot generated with:")
        for key in ['train_loss', 'val_loss', 'learning_rate']:
            if key in self.step_history and self.step_history[key]:
                print(f"  {key}: {len(self.step_history[key])} points")

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