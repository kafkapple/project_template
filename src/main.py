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


# resolve 경로를 표준화하고, 절대 경로를 반환하는 메서드입니다. 이 메서드는 심볼릭 링크를 따라가거나, 상대 경로를 절대 경로로 변환하는 등 경로를 명확하게 정리하