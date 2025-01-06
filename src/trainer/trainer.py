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
                lr=cfg.train.optimizer.lr
            )
        else:
            self.optimizer = None
        
        self.save_dir = Path(cfg.dirs.outputs) / 'checkpoints'
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # 메트릭 계산기 초기화 - cfg 전달
        self.train_metrics = MetricCalculator(cfg, cfg.train.metrics.train)
        self.val_metrics = MetricCalculator(cfg, cfg.train.metrics.val)
        
        # 모델 체크포인터 초기화
        self.checkpointer = ModelCheckpointer(cfg, self.save_dir)
        
        # Scheduler 초기화
        if cfg.train.training.scheduler.type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=cfg.train.training.scheduler.params.step_size,
                gamma=cfg.train.training.scheduler.params.gamma
            )
        elif cfg.train.training.scheduler.type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.train.training.scheduler.params.T_max,
                eta_min=cfg.train.training.scheduler.params.eta_min
            )

    def _train_epoch_torch(self):
        """PyTorch 모델용 학습"""
        self.model.train()
        total_loss = 0
        all_outputs = []
        all_labels = []
        all_images = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, labels) in enumerate(pbar):
            data, labels = data.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            # step 단위 로깅 - 실제 step 번호 사용
            if self.cfg.train.metrics.step.enabled:
                if batch_idx % self.cfg.train.metrics.step.frequency == 0:
                    global_step = (self.current_epoch - 1) * len(self.train_loader) + batch_idx
                    step_metrics = {
                        'loss': loss.item(),
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    }
                    self.train_metrics.add_step_metrics(step_metrics, global_step)
            
            total_loss += loss.item()
            all_outputs.append(outputs.detach().cpu())
            all_labels.append(labels.cpu())
            all_images.append(data.cpu())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 에포크 단위 메트릭 계산
        epoch_outputs = torch.cat(all_outputs)
        epoch_labels = torch.cat(all_labels)
        epoch_images = torch.cat(all_images)
        
        avg_loss = total_loss / len(self.train_loader)
        
        # scheduler step 이후의 learning rate를 사용하기 위해 
        # learning rate는 metrics에서 제외하고 나중에 추가
        metrics = {
            'loss': avg_loss
        }
        
        # train phase 메트릭 계산 및 로깅
        train_phase_metrics = self.train_metrics.calculate(
            epoch_outputs, 
            epoch_labels,
            phase='train',
            step=self.current_epoch,
            logger=self.wandb_logger,
            images=epoch_images,
            loss=avg_loss
        )
        
        metrics.update({
            f"train/{k}": v for k, v in train_phase_metrics.items()
        })
        
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
            loss=total_loss / len(self.val_loader),
            
        )
        metrics['loss'] = total_loss / len(self.val_loader)
        
        return metrics

    def train(self):
        self.current_epoch = 0
        best_metrics = {
            'best_epoch': -1
        }

        for epoch in range(1, self.cfg.train.training.epochs + 1):
            self.current_epoch = epoch
            train_metrics = self._train_epoch_torch()
            val_metrics = self.validate()
            
            # scheduler step 호출
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            
            # scheduler step 이후에 현재 learning rate 추가
            current_lr = self.optimizer.param_groups[0]['lr']
            train_metrics.update({
                'learning_rate': current_lr,
                'train/learning_rate': current_lr
            })
            
            # 메트릭 로깅
            self.wandb_logger.log_metrics({
                **{k: v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
                "epoch": epoch,
            }, phase=None, step=epoch)

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
                
                # best metrics 업데이트 - train.yaml의 val metrics에 따라
                best_metrics.update({
                    'best_epoch': epoch,
                    **{f"best_{k}": v for k, v in val_metrics.items() 
                       if k in self.cfg.train.metrics.val}  # val metrics에 있는 것만 저장
                })
                
                # best model 저장 시점에 summary 업데이트
                self.wandb_logger.run.summary.update(best_metrics)
            
            # 콘솔 출력
            print(
                f"========== Epoch {epoch}: " + 
                ", ".join([f"{k}: {v:.4f}" for k, v in {
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()}
                }.items()])
            )
        
        # 학습 완료 후 step 히스토리 로깅
        self.train_metrics.log_step_history(self.wandb_logger)
        