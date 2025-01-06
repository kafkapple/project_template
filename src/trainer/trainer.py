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
        
        # 메트릭 계산기 초기화 - 동일한 cfg 객체 전달
        self.metrics_calculator = MetricCalculator(cfg, cfg.train.metrics.train + cfg.train.metrics.val)
        
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
            
            # step 단위 로깅 - batch 단위로만 기록
            if self.cfg.train.metrics.step.enabled:
                if batch_idx % self.cfg.train.metrics.step.frequency == 0:
                    global_step = (self.current_epoch - 1) * len(self.train_loader) + batch_idx
                    step_metrics = {
                        'loss': loss.item(),
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    }
                    self.metrics_calculator.add_step_metrics(step_metrics, global_step)
            
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
        
        # 원본 메트릭 계산 - 단일 메트릭 계산기 사용
        current_lr = self.optimizer.param_groups[0]['lr']
        base_metrics = self.metrics_calculator.calculate(
            epoch_outputs, 
            epoch_labels,
            phase='train',
            step=self.current_epoch,
            logger=self.wandb_logger,
            images=epoch_images,
            loss=avg_loss,
            learning_rate=current_lr
        )
        
        # 로깅용 메트릭 생성
        metrics_for_log = {f"train/{k}": v for k, v in base_metrics.items()}
        
        return {
            'metrics': base_metrics,           # 원본 메트릭
            'metrics_for_log': metrics_for_log # 로깅용 메트릭
        }

    def validate(self):
        """검증 수행"""
        self.model.eval()
        total_loss = 0
        all_outputs = []
        all_labels = []
        all_images = []

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
                all_images.append(data)
        
        # 검증 메트릭 계산
        epoch_outputs = torch.cat(all_outputs)
        epoch_labels = torch.cat(all_labels)
        epoch_images = torch.cat(all_images)
        
        val_loss = total_loss / len(self.val_loader)
        
        # validation phase의 step 메트릭 추가
        if self.cfg.train.metrics.step.enabled:
            step_metrics = {'loss': val_loss}
            self.metrics_calculator.add_step_metrics(
                step_metrics, 
                self.current_epoch,
                phase='val'  # validation phase 명시
            )
        
        # 원본 메트릭 계산
        base_metrics = self.metrics_calculator.calculate(
            epoch_outputs, 
            epoch_labels,
            phase='val',
            step=self.current_epoch,
            logger=self.wandb_logger,
            images=epoch_images,
            loss=val_loss,
            learning_rate=None
        )
        
        # 로깅용 메트릭 생성
        metrics_for_log = {f"val/{k}": v for k, v in base_metrics.items()}
        
        return {
            'metrics': base_metrics,           # 원본 메트릭
            'metrics_for_log': metrics_for_log # 로깅용 메트릭
        }

    def train(self):
        self.current_epoch = 0
        best_metrics = {
            'best_epoch': -1
        }

        for epoch in range(1, self.cfg.train.training.epochs + 1):
            self.current_epoch = epoch
            train_results = self._train_epoch_torch()
            val_results = self.validate()
            
            # scheduler step 호출
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            
            # scheduler step 이후에 현재 learning rate 추가
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # train 메트릭에 learning rate 추가
            train_results['metrics']['learning_rate'] = current_lr  # 원본 메트릭에 추가
            train_results['metrics_for_log']['train/learning_rate'] = current_lr  # 로깅용 메트릭에 추가
            
            # 로깅용 메트릭 준비
            combined_metrics = {
                **train_results['metrics_for_log'],
                **val_results['metrics_for_log'],
                'epoch': epoch
            }
            
            # wandb 로깅
            self.wandb_logger.log_epoch_metrics(combined_metrics, epoch)
            
            # step 단위 메트릭 업데이트
            if self.cfg.train.metrics.step.enabled:
                step_metrics = {
                    'loss': train_results['metrics']['loss'],
                    'learning_rate': current_lr  # prefix 없는 버전 사용
                }
                self.metrics_calculator.add_step_metrics(step_metrics, self.current_epoch)
            
            # 콘솔 출력
            print(
                f"========== Epoch {epoch}: " + 
                ", ".join([f"{k}: {v:.4f}" for k, v in {
                    **{f"train_{k}": v for k, v in train_results['metrics'].items()},
                    **{f"val_{k}": v for k, v in val_results['metrics'].items()}
                }.items()])
            )
        
        # 학습 완료 후 step 히스토리 로깅
        self.metrics_calculator.log_step_history(self.wandb_logger)
        