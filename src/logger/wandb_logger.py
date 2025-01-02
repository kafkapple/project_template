import wandb
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
class WandbLogger:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.project_name = cfg.logger.wandb.project_name
        self.entity = cfg.logger.wandb.entity
        self.run = self.init_wandb()
        
    def init_wandb(self):
        try:
            run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=self.cfg.project.timestamp,
                config=OmegaConf.to_container(self.cfg.train, resolve=True), # run column info
                tags=self.cfg.logger.wandb.tags
            )
        except Exception as e:
            print(f'Failed to initialize WandB: {e}')
            return None
        return run
    def log_params(self, params: dict):
        if isinstance(params, DictConfig):
            params = OmegaConf.to_container(params, resolve=True)
        wandb.config.update(params)
    def log_metrics(self, metrics: dict, phase, step=None, fold=None):
        wandb_metrics = {}
        
        # 메트릭 이름에 phase 추가
        for k, v in metrics.items():
            metric_name = f"{phase}/{k}" if phase else k
            if fold is not None:
                metric_name = f"fold_{fold}/{metric_name}"
            wandb_metrics[metric_name] = v
        
        if step is not None:
            wandb_metrics['epoch'] = step
        
        wandb.log(wandb_metrics)
    
    def add_summary(self, summary_dict: dict):
        wandb.summary.update(summary_dict)
  
    def save_model(self, model, filename=None):
        if filename is None:
            filename = f'{self.run_name}.pt'
        filepath = Path(self.save_dir) / filename
        torch.save(model.state_dict(), filepath)
        wandb.save(filepath)
        print(f'Model saved to {filepath}')
    def add_artifact(self, artifact_name: str, artifact_type: str, artifact_path: str):
        wandb.log_artifact(artifact_name, artifact_type, artifact_path)
    def finish(self):
        wandb.finish()
