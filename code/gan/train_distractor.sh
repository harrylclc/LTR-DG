CUDA_VISIBLE_DEVICES=1 python gan.py --dataset ./distractorQA/sciq_vocab_new/\
  --prefix distractorQA_sciq\
  --num_epochs 100\
  --batch_size 64\
  --pools_size 64\
  --gan_k 16\
  --max_sequence_length_q 80\
  --max_sequence_length_a 20\
  --pretrained_embeddings_path ./data_embeddings/glove.840B.300d.txt\
  # --pretrained_model_path ./model/distractorQA_Dis_-1_0.5227108084613155_0.4854253195531082_0.4198782961460446_0.46551724137931055.model
