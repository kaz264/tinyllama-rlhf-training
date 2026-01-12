# TinyLlama RLHF Training Project

PPO(Proximal Policy Optimization)를 사용한 TinyLlama 모델 강화학습 파인튜닝 프로젝트입니다.

## 프로젝트 소개

이 프로젝트는 TinyLlama-1.1B-Chat 모델을 RLHF(Reinforcement Learning from Human Feedback) 기법으로 파인튜닝하여, 긍정적인 감정 표현을 생성하도록 학습시킵니다.

## 주요 특징

- **모델**: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B 파라미터)
- **학습 방법**: PPO (Proximal Policy Optimization)
- **효율적 학습**: LoRA (Low-Rank Adaptation) 적용
- **보상 함수**: 긍정적인 감정 단어 생성 시 보상
- **목표 단어**: happy, glad, good, great, smile, joy, love

## 기술 스택

- **PyTorch**: 딥러닝 프레임워크
- **Transformers**: Hugging Face 트랜스포머 라이브러리
- **TRL**: Transformer Reinforcement Learning 라이브러리
- **PEFT**: Parameter-Efficient Fine-Tuning (LoRA 구현)
- **Accelerate**: 분산 학습 및 최적화
- **bitsandbytes**: 모델 양자화 및 최적화

## 설치 방법

### 1. Python 환경 설정

Python 3.8 이상이 필요합니다.

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. 필수 라이브러리 설치

```bash
pip install -r requirements.txt
```

또는 개별 설치:

```bash
pip install torch transformers==4.40.0 trl==0.8.6 peft==0.10.0 accelerate==0.29.3 bitsandbytes
```

### 3. CUDA 설정 (선택 사항)

NVIDIA GPU를 사용하는 경우, CUDA가 설치되어 있는지 확인하세요. GPU 없이도 CPU로 실행 가능하지만, 학습 속도가 느릴 수 있습니다.

```bash
# CUDA 사용 가능 여부 확인
python -c "import torch; print(torch.cuda.is_available())"
```

## 사용 방법

### 학습 실행

```bash
python train_rlhf.py
```

### 학습 과정

1. TinyLlama 모델과 토크나이저 로드
2. LoRA 설정 적용
3. PPO 트레이너 초기화
4. 100 에포크 동안 학습 수행
5. 학습된 모델을 `./trained_model/` 디렉토리에 저장
6. 테스트 문장으로 결과 확인

### 학습 중 표시되는 정보

- 현재 에포크 및 스텝 번호
- 긍정적인 단어 생성 발견 시 알림
- 10 에포크마다 진행 상황 및 성공률

## 학습 설정

### LoRA 설정

```python
r=16               # LoRA 랭크 (낮을수록 파라미터 적음)
lora_alpha=32      # 학습 강도
lora_dropout=0.05  # 드롭아웃 비율
```

### PPO 설정

```python
learning_rate=2e-5              # 학습률
batch_size=10                   # 배치 크기
mini_batch_size=2               # 미니 배치 크기
gradient_accumulation_steps=5   # 그래디언트 누적 스텝
ppo_epochs=4                    # PPO 에포크 (복습 횟수)
init_kl_coef=0.2               # KL 페널티 계수
```

### 학습 파라미터

- **훈련 에포크**: 100
- **배치 크기**: 10
- **입력 쿼리**: "I feel so", "This makes me", "I am really"
- **최대 생성 토큰**: 10

## 보상 함수

모델이 다음 단어 중 하나를 생성하면 +1.0의 보상을 받습니다:
- happy, glad, good, great, smile, joy, love

이외의 경우 -1.0의 페널티를 받습니다.

## 출력 결과

### 학습 중

```
🔥 [Epoch 1/100] Step 1 학습 완료
🎉 [Epoch 1] 발견! 'happy and excited!'

📊 [진행 상황] Epoch 10/100
   긍정 생성 횟수: 15
   성공률: 50.0%
```

### 학습 완료 후

```
=== 훈련 종료 ===
총 학습 스텝: 30
긍정 생성 횟수: 120
최종 성공률: 40.0%

✅ 모델이 ./trained_model/에 저장되었습니다!

=== 학습된 모델 테스트 ===
입력: 'I feel so'
생성: 'I feel so happy and grateful today'
```

## 모델 저장

학습이 완료되면 모델은 `./trained_model/` 디렉토리에 자동 저장됩니다.

저장된 모델 로드 방법:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./trained_model")
model = AutoModelForCausalLM.from_pretrained("./trained_model")

inputs = tokenizer("I feel so", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0]))
```

## 파일 구조

```
test_sample_dl/
├── train_rlhf.py         # 메인 학습 스크립트
├── requirements.txt      # 의존성 패키지
├── README.md            # 프로젝트 문서
└── trained_model/       # 학습된 모델 저장 디렉토리 (자동 생성)
```

## 문제 해결

### CUDA 메모리 부족

- 배치 크기를 줄이세요: `batch_size=5`
- LoRA 랭크를 낮추세요: `r=8`
- 그래디언트 체크포인팅 활성화

### 학습이 너무 느림

- GPU 사용 여부 확인
- 배치 크기를 늘리세요
- 에포크 수를 줄이세요

### 모델이 긍정 단어를 생성하지 않음

- 학습 에포크를 늘리세요: `TRAINING_EPOCHS = 200`
- 학습률을 조정하세요: `learning_rate=3e-5`
- PPO 에포크를 늘리세요: `ppo_epochs=6`

## 참고 자료

- [TRL Documentation](https://huggingface.co/docs/trl)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [TinyLlama Model](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)

## 라이센스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다.

## 기여

버그 리포트나 개선 제안은 이슈로 등록해주세요!

---

Happy Training! 🚀
