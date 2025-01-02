from pathlib import Path
from omegaconf import DictConfig, OmegaConf
class PrintLogger:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.experiment_data= {}
    def print_info(self):
        print(f'===Project===\n{OmegaConf.to_yaml(self.cfg.project, resolve=True)}') # resolve=True:  ${var}를 실제 값으로 출력
        print(f"===Dirs===\n{OmegaConf.to_yaml(self.cfg.dirs, resolve=True)}")
        print(f'===Logger===\n{OmegaConf.to_yaml(self.cfg.logger, resolve=True)}')
        print(f'===Data===\n{OmegaConf.to_yaml(self.cfg.data, resolve=True)}')
        print(f'===Train===\n{OmegaConf.to_yaml(self.cfg.train, resolve=True)}')
        print(f'===Model===\n{OmegaConf.to_yaml(self.cfg.model, resolve=True)}')
        print(f'===Metrics===\n{OmegaConf.to_yaml(self.cfg.metrics, resolve=True)}')
