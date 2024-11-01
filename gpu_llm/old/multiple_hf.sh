#!/bin/bash

# 모델 리스트
models=("facebook/opt-6.7b")

# (input, output) 길이 리스트
lengths=(
  "20 60"
  "160 320"
)

# 배치 사이즈 리스트
batch_sizes=(1 2 4 8 16 32 64 128 256 512 1024)

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
    error_occurred=false
    
    for batch in "${batch_sizes[@]}"; do
      # 이전에 에러가 발생했으면 다음 모델로 넘어감
      if [ "$error_occurred" = true ]; then
        break
      fi
      
      # 모델 이름에서 슬래시를 언더바로 변환
      model_name=$(echo $model | sed 's/\//_/g')
      
      # dtype이 fp32가 아니면 파일 이름에 dtype 추가
      if [ "$dtype" == "fp32" ]; then
        log_file="./logs/${model_name}_batch_${batch}_in_${I}_out_${O}.txt"
      else
        log_file="./logs/${model_name}_batch_${batch}_in_${I}_out_${O}_${dtype}.txt"
      fi
      
      # 로그 파일이 이미 존재하면 스킵
      if [ -f "$log_file" ]; then
        echo "Skipping: $log_file already exists."
        continue
      fi
      
      # 실행 명령어 및 stdout 저장
      echo "Running: python run_opt.py --model $model --input-length $I --output-length $O --batch-size $batch --dtype $dtype"
      python run_opt.py --model $model --input-length $I --output-length $O --batch-size $batch --dtype $dtype > "$log_file" 2>&1
      
      # 에러 발생 여부 확인
      if [ $? -ne 0 ]; then
        echo "Error encountered with model $model at batch size $batch. Skipping to next model."
        error_occurred=true
        break
      fi
    done
  done
done
