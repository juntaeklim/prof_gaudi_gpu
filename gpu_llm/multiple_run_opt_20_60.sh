#!/bin/bash

# 모델 리스트
models=("facebook/opt-125m" "facebook/opt-1.3b")
# models=("facebook/opt-125m" "facebook/opt-1.3b" "facebook/opt-6.7b")

# 고정된 input/output 길이
I=20
O=60

# 배치 사이즈 리스트
batch_sizes=(1 2 4 8 16 32 64 128 256 512 1024)

# 루프를 돌며 모든 경우에 대해 실행
for model in "${models[@]}"; do
  error_occurred=false
  
  for batch in "${batch_sizes[@]}"; do
    # 이전에 에러가 발생했으면 다음 모델로 넘어감
    if [ "$error_occurred" = true ]; then
      break
    fi
    
    # 모델 이름에서 슬래시를 언더바로 변환
    model_name=$(echo $model | sed 's/\//_/g')
    
    # 출력할 로그 파일 경로 설정
    log_file="./logs/${model_name}_batch_${batch}_in_${I}_out_${O}.txt"
    
    # 로그 파일이 이미 존재하면 스킵
    if [ -f "$log_file" ]; then
      echo "Skipping: $log_file already exists."
      continue
    fi
    
    # 실행 명령어 및 stdout 저장
    echo "Running: python run_opt.py --model $model --input-length $I --output-length $O --batch-size $batch"
    python run_opt.py --model $model --input-length $I --output-length $O --batch-size $batch > "$log_file" 2>&1
    
    # 에러 발생 여부 확인
    if [ $? -ne 0 ]; then
      echo "Error encountered with model $model at batch size $batch. Skipping to next model."
      error_occurred=true
      break
    fi
  done
done
