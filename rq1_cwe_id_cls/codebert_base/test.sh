python bert_base_main.py \
  --output_dir=./saved_models \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --do_test \
  --test_data_file=../../data/test.csv \
  --block_size 512 \
  --eval_batch_size 16 \
  --seed 123456  2>&1 | tee test.log