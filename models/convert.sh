# Using the training checkpoints to convert to pytorch model for testing.
# the paths for --src needs to be changed accordingly.


python convert_dpr_original_checkpoint_to_pytorch.py \
--type question_encoder \
--src ../DPR-main/outputs/2023-04-15/18-54-14/dpr_biencoder.$1 \
--dest question_encoder 


python convert_dpr_original_checkpoint_to_pytorch.py \
--type ctx_encoder \
--src ../DPR-main/outputs/2023-04-15/18-54-14/dpr_biencoder.$1 \
--dest ctx_encoder 

cp pytorch/question_encoder/tokenizer_config.json question_encoder/

cp pytorch/ctx_encoder/tokenizer_config.json ctx_encoder/

cp pytorch/question_encoder/vocab.txt question_encoder/

cp pytorch/ctx_encoder/vocab.txt ctx_encoder/












