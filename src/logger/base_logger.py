from omegaconf import DictConfig
from pathlib import Path
import os
from .wandb_logger import WandbLogger
from .print_logger import PrintLogger
import shutil
class Logger:
    def __init__(self, cfg: DictConfig):
        """
        Initialize Logger.

        Args:
            cfg (DictConfig): Configuration object.

        Setup experiment directories and loggers, and set random seeds.
        """
        self.cfg = cfg
        self.exp_dir = Path(self.cfg.dirs.outputs)
        self.setup_exp_dirs()
        self.setup_loggers()
        self.set_seed(self.cfg.project.seed)
        self._backup_configs()
    def setup_exp_dirs(self):
        """
        실험 디렉토리를 설정. outputs 디렉토리 밑에 subdirs에 있는 디렉토리들을 생성.

        """
        for subdir in self.cfg.dirs.subdirs:
            os.makedirs(Path(self.cfg.dirs.outputs) / subdir, exist_ok=True)
    def get_exp_path(self, category: str) -> Path:
        """특정 카테고리의 실험 경로 반환"""
        return self.exp_dir / category
    def setup_loggers(self):
        """
        Initializes the loggers for the experiment.

        This method sets up the WandbLogger and PrintLogger using the
        provided configuration.

        """
        self.wandb_logger = WandbLogger(self.cfg)
        self.print_logger = PrintLogger(self.cfg)
    def _setup_debug_mode(self):
        """디버그 모드 설정"""
        self.cfg.logger.project_name = "debug_test"
        # 학습 관련 설정 수정
        self.cfg.model.name = "resnet18"
        
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
    def _backup_configs(self):
        """configs 폴더를 실험 디렉토리에 복사"""
        src_config_dir = Path("config")
        dst_config_dir = self.exp_dir / "config"
        print(f"src_config_dir: {src_config_dir}")
        print(f"dst_config_dir: {dst_config_dir}")
        
        if src_config_dir.exists():
            print(f"\nBacking up configs to: {dst_config_dir}")
            shutil.copytree(src_config_dir, dst_config_dir)
        else:
            print("\nWarning: configs directory not found")
    