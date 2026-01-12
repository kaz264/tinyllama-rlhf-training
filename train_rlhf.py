"""
TinyLlama 모델을 PPO(Proximal Policy Optimization)로 파인튜닝하는 스크립트
보상 함수: 긍정적인 감정 단어를 생성하면 보상을 받습니다.
"""

import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig
import os

# ==========================================
# 1. 설정 (Configuration)
# ==========================================

# 모델 설정
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./trained_model"

# LoRA 설정 (거대 모델 전체가 아니라, 이 부분만 학습합니다)
LORA_CONFIG = LoraConfig(
    r=16,               # LoRA 랭크
    lora_alpha=32,      # 학습 강도
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# PPO 설정 (학습률 UP, 복습 UP)
PPO_CONFIG = PPOConfig(
    learning_rate=3e-5,  # 개선된 학습률 (2e-5 -> 3e-5)
    batch_size=10,
    mini_batch_size=2,
    gradient_accumulation_steps=5,
    ppo_epochs=4,        # 같은 데이터로 4번 반복 학습 (복습)
    init_kl_coef=0.2,    # KL 페널티 시작값 (기본값 0.1보다 2배 높임)
    adap_kl_ctrl=True,   # 상황에 맞춰서 페널티 강도를 AI가 자동 조절
)

# 학습 설정
TRAINING_EPOCHS = 150  # 개선된 에포크 수 (100 -> 150)
TARGET_BATCH_SIZE = 10
# 확장된 입력 쿼리 (더 다양한 문장 패턴)
QUERIES = [
    "I feel so",
    "This makes me",
    "I am really",
    "Today I am",
    "Life is",
    "I'm feeling"
]
# 확장된 긍정 단어 목록 (더 다양한 감정 표현)
TARGET_WORDS = [
    "happy", "glad", "good", "great", "smile", "joy", "love",
    "wonderful", "amazing", "excited", "pleased", "delighted",
    "cheerful", "grateful", "blessed", "fantastic"
]

# ==========================================
# 2. 디바이스 설정
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 디바이스: {device}")
if device == "cpu":
    print("⚠️  경고: CUDA를 사용할 수 없습니다. CPU로 학습하면 매우 느릴 수 있습니다.")

# ==========================================
# 3. 모델과 토크나이저 준비
# ==========================================
print(f"\n모델({MODEL_ID})을 불러오는 중입니다... 잠시만 기다려주세요.")

try:
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # TinyLlama는 패딩 토큰이 따로 없어서 설정해줘야 함
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 모델 로드 (LoRA 적용)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        MODEL_ID,
        peft_config=LORA_CONFIG,
        device_map="auto" if device == "cuda" else None
    )

    # PPO 트레이너 생성
    ppo_trainer = PPOTrainer(
        config=PPO_CONFIG,
        model=model,
        tokenizer=tokenizer,
    )

    print("✅ 모델 로드 완료!")

except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    raise

# ==========================================
# 4. 보상 함수 (Reward Function)
# ==========================================
def get_reward(generated_text):
    """
    생성된 텍스트에 긍정적인 감정 단어가 포함되어 있으면 보상을 줍니다.

    Args:
        generated_text (str): 생성된 텍스트

    Returns:
        float: 보상 값 (1.0 또는 -1.0)
    """
    # 하나라도 포함되면 성공!
    for word in TARGET_WORDS:
        if word in generated_text.lower():
            return 1.0

    return -1.0

# ==========================================
# 5. 학습 루프 (Training Loop)
# ==========================================
print("\n=== 훈련 시작 ===")
print(f"훈련 에포크: {TRAINING_EPOCHS}")
print(f"배치 크기: {TARGET_BATCH_SIZE}")
print(f"목표 단어: {', '.join(TARGET_WORDS)}\n")

batch_queries = []
batch_responses = []
batch_rewards = []

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 10
}

total_steps = 0
positive_generations = 0

try:
    for epoch in range(TRAINING_EPOCHS):
        for query_txt in QUERIES:
            # 입력 준비
            inputs = tokenizer(query_txt, return_tensors="pt")
            query_tensors = inputs.input_ids.to(device)

            # 텍스트 생성
            response_tensors = ppo_trainer.generate(query_tensors[0], **generation_kwargs)
            response_txt = tokenizer.decode(response_tensors[0], skip_special_tokens=True)
            generated_part = response_txt[len(query_txt):]

            # 보상 계산
            reward_value = get_reward(generated_part)

            # 배치에 추가
            batch_queries.append(query_tensors[0])
            batch_responses.append(response_tensors[0])
            batch_rewards.append(torch.tensor(reward_value).to(device))

            # 배치가 찼으면 학습 수행
            if len(batch_queries) == TARGET_BATCH_SIZE:
                stats = ppo_trainer.step(batch_queries, batch_responses, batch_rewards)
                total_steps += 1

                # 배치 초기화
                batch_queries = []
                batch_responses = []
                batch_rewards = []

                print(f"🔥 [Epoch {epoch+1}/{TRAINING_EPOCHS}] Step {total_steps} 학습 완료")

            # 긍정적인 생성 결과 표시
            if reward_value > 0:
                positive_generations += 1
                print(f"🎉 [Epoch {epoch+1}] 발견! '{generated_part.strip()}'")

        # 10 에포크마다 진행 상황 출력
        if (epoch + 1) % 10 == 0:
            success_rate = (positive_generations / ((epoch + 1) * len(QUERIES))) * 100
            print(f"\n📊 [진행 상황] Epoch {epoch+1}/{TRAINING_EPOCHS}")
            print(f"   긍정 생성 횟수: {positive_generations}")
            print(f"   성공률: {success_rate:.1f}%\n")

except KeyboardInterrupt:
    print("\n\n⚠️  훈련이 사용자에 의해 중단되었습니다.")
except Exception as e:
    print(f"\n\n❌ 훈련 중 오류 발생: {e}")
    raise

print("\n=== 훈련 종료 ===")
print(f"총 학습 스텝: {total_steps}")
print(f"긍정 생성 횟수: {positive_generations}")
print(f"최종 성공률: {(positive_generations / (TRAINING_EPOCHS * len(QUERIES))) * 100:.1f}%")

# ==========================================
# 6. 모델 저장
# ==========================================
print(f"\n모델을 저장하는 중... ({OUTPUT_DIR})")
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ 모델이 {OUTPUT_DIR}에 저장되었습니다!")
except Exception as e:
    print(f"❌ 모델 저장 실패: {e}")

# ==========================================
# 7. 테스트 생성
# ==========================================
print("\n=== 학습된 모델 테스트 ===")
test_queries = ["I feel so", "This makes me", "Today I am"]

for test_query in test_queries:
    inputs = tokenizer(test_query, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=15,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"입력: '{test_query}'")
    print(f"생성: '{generated_text}'\n")

print("완료! 🎉")
