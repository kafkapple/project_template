import logging
import sys
from pathlib import Path
from abc import ABC, abstractmethod

class BaseLogger(ABC):
    """로거 기본 클래스"""
    def __init__(self, cfg):
        self.cfg = cfg
        self.setup_file_logging()
        
    def setup_file_logging(self):
        """파일 로깅 설정"""
        log_dir = Path(self.cfg.dirs.outputs) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_dir / "train.log")
            ]
        )
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def log_metrics(self, metrics: dict, step: int = None):
        """메트릭 로깅 인터페이스"""
        pass

    def info(self, msg: str):
        """일반 로그 메시지"""
        self.logger.info(msg)

    def warning(self, msg: str):
        """경고 메시지"""
        self.logger.warning(msg)

    def error(self, msg: str):
        """에러 메시지"""
        self.logger.error(msg)