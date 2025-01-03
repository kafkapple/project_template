# Deep Learning Project Template with Hydra and WandB Integration

## 개요
- 딥러닝 실험을 위한 모듈화된 템플릿
  - Hydra 설정 관리
  - WandB(Weights & Biases)를 통한 실험 관리 및 시각화 기능

## 주요 기능
- **모듈화된 구조**: 데이터, 모델, 학습 로직이 명확하게 분리된 구조
- **Hydra 설정 관리**: 
  - 모듈화된 설정 파일
  - 커맨드 라인을 통한 손쉬운 설정 변경
  - 실험별 설정 자동 백업
- **다양한 모델 지원**: 
  - EfficientNet (timm)
  - Vision Transformer (ViT)
  - DeiT (Data-efficient Image Transformer)
  - MLP
- **다양한 데이터셋 지원**:
  - MNIST
  - Fashion MNIST
  - CIFAR-10
  - SVHN
- **WandB 통합**:
  - 실험 설정 자동 로깅
  - 메트릭 실시간 모니터링
  - 학습 곡선 시각화
  - 예측 샘플 시각화
  - Confusion Matrix
  - PR Curve

## 프로젝트 구조
```
project/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # 기본 설정 (프로젝트, 경로, WandB 설정)
│   ├── data/                  # 데이터셋 설정
│   │   ├── mnist.yaml
│   │   ├── fashion_mnist.yaml
│   │   ├── cifar10.yaml
│   │   ├── svhn.yaml
│   │   └── dummy.yaml
│   ├── model/                 # 모델 설정
│   │   ├── efficientnet.yaml
│   │   ├── vit.yaml
│   │   ├── deit.yaml
│   │   └── mlp.yaml
│   └── train/                 # 학습 설정
│       └── train.yaml         # 학습 하이퍼파라미터, 메트릭 설정
├── src/
│   ├── data/                  # 데이터 관련 모듈
│   │   ├── __init__.py
│   │   ├── datasets.py       # 데이터셋 Factory
│   │   └── dataloader.py     # DataLoader 생성
│   ├── models/               # 모델 관련 모듈
│   │   ├── __init__.py
│   │   ├── models.py       # 모델 정의
│   │   └── model_factory.py  # 모델 Factory
│   ├── train/               # 학습 관련 모듈
│   │   ├── __init__.py
│   │   ├── train.py        # Trainer 클래스
│   │   ├── metrics.py      # 메트릭 계산 및 로깅
│   ├── logger/             # 로깅 관련 모듈
│   │   ├── __init__.py
│   │   ├── base_logger.py  # 기본 로거
│   │   └── wandb_logger.py # WandB 로거
│   └── main.py            # 메인 실행 파일
├── environment.yml        # Conda 환경 설정
└── .gitignore
```

## 설치 방법

### 환경 설정
```bash
# 1. 저장소 클론
git clone <repository-url>
cd project_template

# 2. Conda 환경 생성
conda env create -f environment.yml
conda activate wandb_test
```

### WandB 설정
1. WandB 계정 생성: https://wandb.ai
2. WandB 로그인
```bash
wandb login
```

## 실행 방법

### 기본 실행
```bash
python src/main.py
```

기본 설정:
- 모델: EfficientNet-B0 (pretrained)
- 데이터셋: CIFAR-10
- 학습 에폭: 2
- 학습률: 0.001
- 배치 크기: 32

### 설정 변경
```bash
# 데이터셋 변경
python src/main.py data=mnist
python src/main.py data=fashion_mnist
python src/main.py data=cifar10
python src/main.py data=svhn

# 모델 변경
python src/main.py model=efficientnet
python src/main.py model=vit
python src/main.py model=deit
python src/main.py model=mlp

# 하이퍼파라미터 조정
python src/main.py train.lr=0.001 train.epochs=10 data.batch_size=64
```

## 주요 기능 설명

### 데이터 처리
- 모든 이미지는 자동으로 224x224 크기로 리사이즈
- MNIST/Fashion-MNIST는 3채널로 자동 변환
- 데이터셋은 train/validation으로 자동 분할 (기본 8:2)

### 모델
- timm 라이브러리 기반의 pretrained 모델 지원
- 커스텀 MLP 모델 지원:
  - 설정 가능한 히든 레이어 차원
  - 드롭아웃 지원
  - ReLU 활성화 함수

### WandB 통합
- 실험 설정 자동 로깅
  - 모델 타입
  - 데이터셋
  - 배치 크기
  - 학습률
  - 타겟 메트릭 (기본: val/f1)
- 실시간 메트릭 모니터링
  - Loss
  - Accuracy
  - F1 Score
  - Precision
  - Recall
- 시각화
  - Learning curves (train/val)
  - Confusion Matrix
  - PR Curve
  - 클래스별 예측 샘플

### 출력 구조
```
outputs/YYYYMMDD_HHMMSS/  # 실험 타임스탬프
├── checkpoints/         # 모델 체크포인트
├── configs/            # 실험 설정 백업
└── logs/               # 로그 파일
    ├── hydra/         # Hydra 로그
    └── wandb/         # WandB 로그
```

## 환경
- Python 3.10
- PyTorch
- timm (Transformer/CNN 모델)
- Hydra (설정 관리)
- WandB (실험 관리)
- scikit-learn (메트릭 계산)
- matplotlib, seaborn (시각화)

## 라이선스
MIT License
