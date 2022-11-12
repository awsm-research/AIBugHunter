python bert_base_main.py \
  --model_name=roberta_base \
  --output_dir=./saved_models \
  --model_type=roberta \
  --tokenizer_name=roberta-base \
  --model_name_or_path=roberta-base \
  --do_train \
  --train_data_file=../../data/train.csv \
  --eval_data_file=../../data/val.csv \
  --test_data_file=../../data/test.csv \
  --epochs 20 \
  --block_size 512 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed 123456  2>&1 | tee train_roberta_base.log