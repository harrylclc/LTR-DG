# train
python ./dataConvertDistractor.py \
  --json ../../data/mcql_processed/train_neg.json \
  --output ../../data/mcql_processed/train.data \
  --vocab_path ../../data/mcql_processed/vocab.txt

# val
python ./dataConvertDistractor.py \
  --json ../../data/mcql_processed/valid_neg.json \
  --output ../../data/mcql_processed/valid.data \
  --vocab_path ../../data/mcql_processed/vocab.txt \
  --train 0

# test 
python ./dataConvertDistractor.py \
  --json ../../data/mcql_processed/test_neg.json \
  --output ../../data/mcql_processed/test.data \
  --vocab_path ../../data/mcql_processed/vocab.txt \
  --train 0
