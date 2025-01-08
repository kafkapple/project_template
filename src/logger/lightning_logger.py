from .base_logger import BaseLogger
from pytorch_lightning.loggers import WandbLogger as PLWandbLogger
import pytorch_lightning as pl
import wandb
import shutil
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

class LightningLogger(BaseLogger):
    """PyTorch Lightning WandbLogger wrapper"""
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._setup_experiment()
        self._setup_debug_mode()
        self.wandb_logger = self._init_wandb_logger()
        
    def _setup_experiment(self):
        """실험 초기화 작업"""
        # 시드 설정
        pl.seed_everything(self.cfg.project.seed, workers=True)
        print(f"\nSet random seed to {self.cfg.project.seed}")
        
        # 설정 출력
        self.print_config()
        
        # 설정 백업
        self._backup_configs()
        
    def print_config(self):
        """설정 정보 출력"""
        print("\n=== Configuration ===")
        print(f'Project:\n{OmegaConf.to_yaml(self.cfg.project, resolve=True)}')
        print(f'Dirs:\n{OmegaConf.to_yaml(self.cfg.dirs, resolve=True)}')
        print(f'Logger:\n{OmegaConf.to_yaml(self.cfg.logger, resolve=True)}')
        print(f'Data:\n{OmegaConf.to_yaml(self.cfg.data, resolve=True)}')
        print(f'Model:\n{OmegaConf.to_yaml(self.cfg.model, resolve=True)}')
        print(f'Train:\n{OmegaConf.to_yaml(self.cfg.train, resolve=True)}')
        print("===================\n")
    
    def _backup_configs(self):
        """configs 폴더를 실험 디렉토리에 복사"""
        src_config_dir = Path("configs")
        dst_config_dir = Path(self.cfg.dirs.outputs) / "configs"
        
        if src_config_dir.exists():
            print(f"\nBacking up configs to: {dst_config_dir}")
            shutil.copytree(src_config_dir, dst_config_dir)
        else:
            print("\nWarning: configs directory not found")

    def _setup_debug_mode(self):
        """Debug 모드 설정은 유지"""
        debug_mode = self.cfg.logger.wandb.job_type == "debug"
        if debug_mode:
            print("Debug mode enabled (from job_type)")
            self.cfg.train.training.epochs = self.cfg.debug.epochs
            self.cfg.train.training.batch_size = self.cfg.debug.batch_size
            self.cfg.data.num_workers = self.cfg.debug.num_workers
            
            if hasattr(self.cfg.data, 'dataset_size'):
                self.cfg.data.dataset_size = int(self.cfg.data.dataset_size * self.cfg.debug.data_ratio)

    def _init_wandb_logger(self):
        """Lightning WandbLogger 초기화"""
        config = {
            "model": self.cfg.model.type,
            "dataset": self.cfg.data.name,
            "target_metric": self.cfg.train.best_model.metric,
            "mode": self.cfg.train.best_model.mode,
            "training": OmegaConf.to_container(self.cfg.train.training),
            "optimizer": OmegaConf.to_container(self.cfg.train.optimizer),
            "regularization": OmegaConf.to_container(self.cfg.train.regularization),
        }
        
        return PLWandbLogger(
            project=self.cfg.logger.wandb.project_name,
            entity=self.cfg.logger.wandb.entity,
            name=f"{self.cfg.project.timestamp}",
            save_dir=str(Path(self.cfg.logger.wandb.dir)),
            config=config,
            job_type=self.cfg.logger.wandb.job_type,
        )

    def experiment(self):
        """wandb.run에 대한 접근자"""
        return self.wandb_logger.experiment

    def watch(self, model, **kwargs):
        """모델 파라미터 로깅"""
        self.wandb_logger.watch(model, **kwargs) 

    def log_metrics(self, metrics: dict, phase: str = None, step: int = None):
        """메트릭 로깅 구현"""
        # phase가 있는 경우 트릭 키에 prefix 추가
        if phase:
            metrics = {f"{phase}/{k}": v for k, v in metrics.items()}
        
        # WandB 로깅
        self.wandb_logger.log_metrics(metrics, step=step)
        
        # 콘솔/파일 로깅을 위한 포맷팅
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        if phase:
            self.info(f"{phase.upper()} - Step {step}: {metrics_str}" if step else f"{phase.upper()}: {metrics_str}")
        else:
            self.info(f"Step {step}: {metrics_str}" if step else metrics_str)

    def log_epoch_metrics(self, train_metrics: dict, val_metrics: dict, epoch: int):
        """에포크 단위 트릭 로깅"""
        combined_metrics = {
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
            'epoch': epoch
        }
        self.wandb_logger.log_metrics(combined_metrics, step=epoch)
        
        # 콘솔 출력용 포맷팅
        train_str = ", ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
        val_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
        self.info(f"Epoch {epoch} - Train: {train_str} | Val: {val_str}")

    def log_step_metrics(self, metrics: dict, step: int, phase: str = 'train'):
        """스텝 단위 메트릭 로깅"""
        step_metrics = {f"{phase}/{k}": v for k, v in metrics.items()}
        self.wandb_logger.log_metrics(step_metrics, step=step)
        
        # 콘솔 출력
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"{phase.upper()} Step {step}: {metrics_str}") 