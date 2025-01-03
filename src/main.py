import hydra
from omegaconf import DictConfig

from logger import Logger
from data import get_dataloaders
from models import ModelFactory
from train import Trainer

@hydra.main(version_base="1.2", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # 로거 초기화
    logger = Logger(cfg)
    
    # 데이터 로더 생성
    train_loader, val_loader = get_dataloaders(cfg)
    
    # 모델 생성
    model = ModelFactory.create_model(cfg)
    
    # 학습
    trainer = Trainer(model, train_loader, val_loader, cfg, logger.wandb_logger)
    trainer.train()
    
    # wandb 종료
    logger.wandb_logger.finish()

if __name__ == "__main__":
    main()

