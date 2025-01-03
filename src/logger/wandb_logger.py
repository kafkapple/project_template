import wandb
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

class WandbLogger:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.project_name = cfg.logger.wandb.project_name
        self.entity = cfg.logger.wandb.entity
        self.run = None
        self.init_wandb()
        
    def init_wandb(self):
        try:
            wandb_dir = Path(self.cfg.logger.wandb.dir)
            wandb_dir.mkdir(parents=True, exist_ok=True)

            # 가장 중요한 설정값들만 config에 포함
            config = {
                "model": self.cfg.model.type,
                "dataset": self.cfg.data.name,
                "batch_size": self.cfg.data.batch_size,
                "learning_rate": self.cfg.train.lr,
                "optimizer": "Adam",
                "target_metric": self.cfg.train.best_model.metric,  # 'val/f1'
                "mode": self.cfg.train.best_model.mode,            # 'max'
                "epochs": self.cfg.train.epochs
            }

            self.run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=f"{self.cfg.project.timestamp}",
                dir=str(wandb_dir),
                config=config,  # 핵심 설정값들
                job_type=self.cfg.logger.wandb.job_type,
                reinit=self.cfg.logger.wandb.reinit
            )
            
            print(f'WandB run initialized with name: {self.run.name}')
        except Exception as e:
            print(f'Failed to initialize WandB: {e}')
            raise e

    def log_params(self, params: dict):
        if isinstance(params, DictConfig):
            params = OmegaConf.to_container(params, resolve=True)
        self.run.config.update(params)

    def log(self, metrics: dict, step=None):
        """wandb.log() 직접 호출"""
        if self.run is not None:
            self.run.log(metrics, step=step)
    
    def log_metrics(self, metrics: dict, phase, step=None, fold=None):
        """기존 메트릭 로깅 메서드"""
        if self.run is None:
            print("Warning: WandB run not initialized")
            return

        wandb_metrics = {}
        for k, v in metrics.items():
            metric_name = f"{phase}/{k}" if phase else k
            if fold is not None:
                metric_name = f"fold_{fold}/{metric_name}"
            wandb_metrics[metric_name] = v
        
        if step is not None:
            wandb_metrics['epoch'] = step
        
        self.run.log(wandb_metrics)
    
    def add_summary(self, summary_dict: dict):
        """WandB summary에 메트릭을 기록합니다."""
        if self.run is not None:
            # summary dict를 한 번에 업데이트
            self.run.summary.update(summary_dict)
            
            # 또는 개별적으로 업데이트할 경우:
            # for key, value in summary_dict.items():
            #     setattr(self.run.summary, key, value)
            # self.run.summary.update()
  
    def save_model(self, model, filename=None):
        if self.run is None:
            return
            
        if filename is None:
            filename = f'{self.run.name}.pt'
        filepath = Path(self.save_dir) / filename
        torch.save(model.state_dict(), filepath)
        self.run.save(filepath)
        print(f'Model saved to {filepath}')

    def add_artifact(self, artifact_name: str, artifact_type: str, artifact_path: str):
        if self.run is not None:
            self.run.log_artifact(artifact_name, artifact_type, artifact_path)

    def finish(self):
        if self.run is not None:
            self.run.finish()
