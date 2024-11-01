#!/bin/bash

# 모델 리스트
models=("meta-llama/Llama-3.1-8B-Instruct")

# (input, output) 길이 리스트
lengths=(
  "756 200"
  "161 388"
  "19 58"
)

# 배치 사이즈 리스트
batch_sizes=(1 2 4 8 16 32 64 128)

# dtype 인자 처리
dtype=$1

# dtype 값이 없거나 올바르지 않으면 에러 메시지 출력 후 종료
if [[ "$dtype" != "fp16" && "$dtype" != "fp32" && "$dtype" != "bf16" ]]; then
  echo "Error: Invalid dtype. Valid values are: fp16, fp32, bf16."
  exit 1
fi

# 루프를 돌며 모든 경우에 대해 실행
for length in "${lengths[@]}"; do
  # I와 O 값을 설정
  I=$(echo $length | cut -d' ' -f1)
  O=$(echo $length | cut -d' ' -f2)

  for model in "${models[@]}"; do
    python parser.py --model $model --input-length $I --output-length $O --dtype $dtype
  done
done
