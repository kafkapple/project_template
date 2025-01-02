import logging
import sys
import os
import shutil
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from .wandb_logger import WandbLogger

class Logger:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.setup_loggers()
      
    def setup_exp_dirs(self):
        """실험 결과 저장을 위한 디렉토리 생성"""
        # 기본 출력 디렉토리 생성
        output_dir = Path(self.cfg.dirs.outputs)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 하위 디렉토리 생성
        for subdir in self.cfg.dirs.subdirs:
            (output_dir / subdir).mkdir(parents=True, exist_ok=True)
            
        print(f"Created experiment directories at: {output_dir}")
    
    def set_seed(self, seed: int):
        """모든 random seed를 고정"""
        import random
        import numpy as np
        import torch
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        print(f"Set random seed to {seed}")
    
    def setup_loggers(self):
        """로거 초기화"""
        # 기본 로거 설정
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(
                    Path(self.cfg.dirs.outputs) / "logs" / "train.log"
                )
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # WandB 로거 초기화
        self.wandb_logger = WandbLogger(self.cfg)
        self.setup_exp_dirs()
        self.print_config()  # 설정 출력 
        self._backup_configs()  # 설정 백업 
        self.set_seed(self.cfg.project.seed)  # 시드 설정      
        
        self.logger.info("Loggers initialized")
    
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
    
    def log_metrics(self, metrics: dict, phase: str = None, step: int = None):
        """메트릭 로깅"""
        # WandB에 로깅
        self.wandb_logger.log_metrics(metrics, phase, step)
        
        # 콘솔에 로깅
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        if phase:
            self.logger.info(f"{phase} - {metric_str}")
        else:
            self.logger.info(metric_str)