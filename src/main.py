
import hydra
from omegaconf import DictConfig
from logger.base_logger import Logger
from metric import metric_test
@hydra.main(version_base="1.2", config_path='../config', config_name='config.yaml')
def main(cfg: DictConfig):
    
    logger = Logger(cfg)
    logger.print_logger.print_info()
    wandb_logger = logger.wandb_logger
    wandb_logger.log_params(cfg.train)

    phases =['train', 'val', 'test']
    for phase in phases:
        for step in range(cfg.train.epochs):
            wandb_logger.log_metrics(metric_test(), phase, step = step + 1)
        wandb_logger.add_summary(metric_test())

if __name__ == '__main__':
    main()


# resolve 경로를 표준화하고, 절대 경로를 반환하는 메서드입니다. 이 메서드는 심볼릭 링크를 따라가거나, 상대 경로를 절대 경로로 변환하는 등 경로를 명확하게 정리하