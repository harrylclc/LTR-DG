CUDA_VISIBLE_DEVICES=0 python gan.py --dataset ../../data/mcql_processed/ \
  --prefix distractorQA\
  --num_epochs 0\
  --batch_size 512\
  --pools_size 512\
  --gan_k 16\
  --max_sequence_length_q 80\
  --max_sequence_length_a 20\
  --pretrained_embeddings_path ./data_embeddings/glove.840B.300d.txt\
