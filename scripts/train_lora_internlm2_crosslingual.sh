#!/bin/bash

# Wait until all GPUs are available and previous run.py is finished
# Feel free to comment out the following block 
while true; do
   echo "Checking GPU memory usage and run.py process..."
   all_available=true

   # Check GPU memory usage
   for i in {0..7}; do
      mem_usage=$(nvidia-smi --id=$i --query-gpu=memory.used --format=csv,noheader,nounits)
      if [ $mem_usage -gt 1000 ]; then
         echo "GPU $i is not available, memory usage: $mem_usage"
         all_available=false
      fi
   done

   # Check for run.py process
   if pgrep -f "run.py" > /dev/null; then
       echo "run.py is currently running."
       all_available=false
   fi

   # Break loop if all conditions are met
   if [ "$all_available" = true ]; then
      echo "All GPUs are available and run.py is not running. Start training..."
      break
   fi

   echo "$(date '+%Y-%m-%d %H:%M:%S')"

   # Wait for 10 minutes before next check
   sleep 600
done

set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/root/.cache/huggingface

port=$(shuf -i25000-30000 -n1)

Seeds=(23 32 33 42)

for SEED in "${Seeds[@]}"; do

   RUN_NAME="crosslingual+multiconer22+bs128+${SEED}"
   # larger GRAD_ACC for larger data
   GRAD_ACC=4
   OUTPUT_DIR="output/MLV2-InternLM2-0s/${RUN_NAME}"
   MODEL_NAME_OR_PATH="/mnt/data/user/yang_yuming/data/Models/internlm2-7b"
   TASK_CONFIG_DIR="configs/ml_configs/crosslingual_multiconer22"
   INSTRUCTION_CONFIG="configs/instruction_config_ml.json"
   DATA_DIR="/mnt/data/user/yang_yuming/data/Public/B2NERD"
   DS_CONFIG="configs/ds_configs/stage2_no_offload.config"
   INPUT_RECORD="${OUTPUT_DIR}/sample_data.record"

   mkdir -p "${OUTPUT_DIR}"
   LOG_FILE="${OUTPUT_DIR}/log.txt"
   exec > >(tee -a "${LOG_FILE}") 2>&1

   # 注意 bs_per_gpu * num_gpu * gradient_accumulation_steps
   # 注意修改 model_name_or_path，output_dir, run_name, data_dir, task_config_dir, instruction_file

   # A800 * 8
   if [ ! -f "$OUTPUT_DIR/train_results.json" ]; then
      echo "Running training on $OUTPUT_DIR"
      deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port $port src/run.py \
         --do_train \
         --do_predict \
         --predict_with_generate \
         --predict_each_epoch \
         --lang auto \
         --model_name_or_path $MODEL_NAME_OR_PATH \
         --data_dir $DATA_DIR \
         --task_config_dir $TASK_CONFIG_DIR \
         --instruction_file $INSTRUCTION_CONFIG \
         --instruction_strategy single \
         --output_dir $OUTPUT_DIR \
         --input_record_file $INPUT_RECORD \
         --bf16 True \
         --seed $SEED \
         --per_device_train_batch_size 4 \
         --per_device_eval_batch_size 4 \
         --gradient_accumulation_steps $GRAD_ACC \
         --gradient_checkpointing True \
         --learning_rate 3e-04 \
         --adam_beta1 0.9 \
         --adam_beta2 0.98 \
         --weight_decay 1e-4 \
         --warmup_ratio 0.02 \
         --lr_scheduler_type "cosine" \
         --adam_epsilon 1e-8 \
         --num_train_epochs 6 \
         --deepspeed $DS_CONFIG \
         --run_name $RUN_NAME \
         --max_source_length 4096 \
         --max_target_length 1024 \
         --generation_max_length 1024 \
         --max_num_instances_per_task 50000 \
         --max_num_instances_per_eval_task 50000 \
         --add_task_name False \
         --add_dataset_name False \
         --num_examples 0 \
         --num_examples_test 0 \
         --train_0shot_prop 1 \
         --train_fewshot_prop 0 \
         --overwrite_output_dir \
         --overwrite_cache \
         --logging_strategy steps \
         --logging_steps 50 \
         --evaluation_strategy epoch \
         --eval_steps 2000 \
         --save_strategy epoch \
         --save_steps 2000 \
         --report_to "none" \
         --log_level info \
         --use_lora True \
         --dynamic_range True \
         --droplabel_rate 0.10 \
         --label_shuffle True
   fi

   ORI_OUTPUT_DIR=$OUTPUT_DIR
   cd src
   python calculate_f1.py --root "../${ORI_OUTPUT_DIR}"
   cd ..
done