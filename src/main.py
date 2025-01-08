import hydra
from omegaconf import DictConfig
from pathlib import Path

from logger import LightningLogger
from data import get_dataloaders
from models import ModelFactory
from trainer import Trainer
from visualization.metric_visualizer import MetricVisualizer

@hydra.main(version_base="1.2", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Lightning Logger 초기화
    logger = LightningLogger(cfg)
    visualizer = MetricVisualizer(Path(cfg.dirs.outputs) / 'metrics')
    
    # 데이터 로더 생성
    train_loader, val_loader = get_dataloaders(cfg)
    
    # 모델 생성
    model = ModelFactory.create_model(cfg)
    logger.watch(model)  # 모델 파라미터 모니터링
    
    # 학습
    trainer = Trainer(
        model, 
        train_loader, 
        val_loader, 
        cfg, 
        logger.wandb_logger,
        visualizer
    )
    trainer.train()
    
    # wandb 종료
    logger.wandb_logger.finish()

if __name__ == "__main__":
    main()

