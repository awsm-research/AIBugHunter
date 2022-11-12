python mobert_main.py \
  --output_dir=./saved_models \
  --model_name=mo_model.bin \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --do_test \
  --test_data_file=../../data/test_wt_type.csv \
  --eval_batch_size 8 \
  --seed 123456  2>&1 | tee test.log