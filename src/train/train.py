import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from pathlib import Path
from .metrics import MetricCalculator, ModelCheckpointer
from sklearn.metrics import classification_report

class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg, wandb_logger):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.wandb_logger = wandb_logger
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nAvailable device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Using: {self.device}\n")
        
        self.model = self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        
        # PyTorch 모델인 경우에만 optimizer 생성
        if isinstance(self.model, nn.Module):
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=cfg.train.lr
            )
        else:
            self.optimizer = None
        
        self.save_dir = Path(cfg.train.save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # 메트릭 계산기 초기화
        self.train_metrics = MetricCalculator(cfg.train.metrics.train)
        self.val_metrics = MetricCalculator(cfg.train.metrics.val)
        
        # 모델 체크포인터 초기화
        self.checkpointer = ModelCheckpointer(cfg, self.save_dir)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_outputs = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc='Training')
        for data, labels in pbar:
            data, labels = data.to(self.device), labels.to(self.device)
            
            if isinstance(self.model, nn.Module):
                # PyTorch 모델 학습
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            else:
                # sklearn/xgboost 모델 학습
                if isinstance(data, torch.Tensor):
                    data = data.cpu().numpy()
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy()
                self.model.fit(data, labels)
                outputs = torch.from_numpy(
                    self.model.predict_proba(data)
                ).float().to(self.device)
                loss = self.criterion(outputs, torch.tensor(labels, device=self.device))

            total_loss += loss.item()
            all_outputs.append(outputs)
            all_labels.append(labels if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=self.device))
            
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})
        
        # 에포크 단위로 메트릭 계산
        epoch_outputs = torch.cat(all_outputs)
        epoch_labels = torch.cat(all_labels)
        metrics = self.train_metrics.calculate(
            epoch_outputs, 
            epoch_labels,
            phase='train',
            step=self.current_epoch,
            logger=self.wandb_logger,
            loss=total_loss / len(self.train_loader)
        )
        metrics['loss'] = total_loss / len(self.train_loader)
        
        # 자세한 분류 리포트 출력 (옵션)
        if self.cfg.train.get('verbose_metrics', False):
            _, predicted = epoch_outputs.max(1)
            report = classification_report(
                self._to_numpy(epoch_labels),
                self._to_numpy(predicted)
            )
            print("\nTraining Classification Report:")
            print(report)
        
        return metrics

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_outputs = []
        all_labels = []
        all_images = []  # 이미지 저장을 위한 리스트 추가

        with torch.no_grad():
            for data, labels in self.val_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                if isinstance(self.model, nn.Module):
                    outputs = self.model(data)
                else:
                    if isinstance(data, torch.Tensor):
                        data = data.cpu().numpy()
                    outputs = torch.from_numpy(
                        self.model.predict_proba(data)
                    ).float().to(self.device)
                
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                all_outputs.append(outputs)
                all_labels.append(labels)
                all_images.append(data)  # 이미지 저장
        
        # 검증 메트릭 계산 (wandb 로깅 포함)
        epoch_outputs = torch.cat(all_outputs)
        epoch_labels = torch.cat(all_labels)
        epoch_images = torch.cat(all_images)  # 이미지 데이터 결합
        
        metrics = self.val_metrics.calculate(
            epoch_outputs, 
            epoch_labels,
            phase='val',
            step=self.current_epoch,
            logger=self.wandb_logger,
            images=epoch_images,
            loss=total_loss / len(self.val_loader)
        )
        metrics['loss'] = total_loss / len(self.val_loader)
        
        return metrics

    def train(self):
        self.current_epoch = 0  # epoch 추적을 위한 변수 추가
        best_metrics = {
            'best_epoch': -1,
            'best_score': float('-inf') if self.checkpointer.mode == 'max' else float('inf')
        }

        # 메트릭 최대/최소값 추적을 위한 딕셔너리
        metric_tracker = {
            'val/f1_max': float('-inf'),
            'val/accuracy_max': float('-inf'),
            'val/loss_min': float('inf')
        }

        for epoch in range(1, self.cfg.train.epochs + 1):
            self.current_epoch = epoch  # 현재 epoch 업데이트
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            # 매 스텝마다의 메트릭 로깅
            self.wandb_logger.log_metrics({
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
                "epoch": epoch,
            }, phase=None, step=epoch)

            # 모트릭 로깅
            metrics = {
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
                "epoch": epoch,
                "best_metric_value": self.checkpointer.best_value,  # 현재까지의 best value
                "target_metric": val_metrics[self.checkpointer.metric_name]  # 현재 target metric
            }
            
            # 최대/최소값 업데이트
            metric_tracker['val/f1_max'] = max(metric_tracker['val/f1_max'], val_metrics['f1'])
            metric_tracker['val/accuracy_max'] = max(metric_tracker['val/accuracy_max'], val_metrics['accuracy'])
            metric_tracker['val/loss_min'] = min(metric_tracker['val/loss_min'], val_metrics['loss'])
            
            # 현재 최대/최소값도 함께 로깅
            metrics.update(metric_tracker)
            
            # best model 저장 및 best metrics 업데이트
            if self.checkpointer.is_better(val_metrics):
                checkpoint_path = self.save_dir / 'best_model.pth'
                
                # PyTorch 모델인 경우
                if isinstance(self.model, nn.Module):
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                        'metrics': val_metrics,
                    }
                    torch.save(checkpoint, checkpoint_path)
                # sklearn/xgboost 모델인 경우
                else:
                    import joblib
                    joblib.dump(self.model.model, checkpoint_path)
                
                best_metrics.update({
                    'best_epoch': epoch,
                    'best_score': val_metrics[self.checkpointer.metric_name],
                    **{f"best_{k}": v for k, v in val_metrics.items()}
                })
                
                # best model 저장 시점에 summary 업데이트
                self.wandb_logger.run.summary.update({
                    "best_epoch": epoch,
                    "best_score": val_metrics[self.checkpointer.metric_name],
                })
            
            # 로깅 - print로 변경
            print(
                f"========== Epoch {epoch}: " + 
                ", ".join([f"{k}: {v:.4f}" for k, v in {
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()}
                }.items()])
            )
        
        # 학습 완료 후 최종 메트릭 기록
        final_summary = {
            'final_epoch': self.cfg.train.epochs - 1,
            'total_epochs': self.cfg.train.epochs,
            **{f"final_{k}": v for k, v in val_metrics.items()},
            **metric_tracker,  # 최대/최소값
            **best_metrics    # best metrics
        }
        self.wandb_logger.add_summary(final_summary)

    def save_checkpoint(self, metrics, epoch):
        if self.checkpointer.is_better(metrics):
            checkpoint_path = self.save_dir / 'best_model.pth'
            
            # PyTorch 모델인 경우
            if isinstance(self.model, nn.Module):
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                    'metrics': metrics,
                }
                torch.save(checkpoint, checkpoint_path)
            # sklearn/xgboost 모델인 경우
            else:
                import joblib
                joblib.dump(self.model.model, checkpoint_path)
            
            # WandB에 아티팩트로 저장
            artifact = wandb.Artifact(
                name=f"model-{self.wandb_logger.run.id}", 
                type="model",
                metadata={
                    'epoch': epoch,
                    **{k: v for k, v in metrics.items() if k in ['f1', 'accuracy']}
                }
            )
            artifact.add_file(str(checkpoint_path))
            self.wandb_logger.run.log_artifact(artifact)
  