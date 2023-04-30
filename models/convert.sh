# Using the training checkpoints to convert to pytorch model for testing.
# the paths for --src needs to be changed accordingly.


#--src ../DPR-main/outputs/2023-04-12/11-16-24/dpr_biencoder.$1 \
#--src ../DPR-main/outputs/2023-04-11/10-17-57/dpr_biencoder.$1 \
python convert_dpr_original_checkpoint_to_pytorch.py \
--type question_encoder \
--src ../DPR-main/outputs/2023-04-15/18-54-14/dpr_biencoder.$1 \
--dest question_encoder 

#--src ../DPR-main/outputs/2023-04-12/11-16-24/dpr_biencoder.$1 \
#--src ../DPR-main/outputs/2023-04-11/10-17-57/dpr_biencoder.$1 \
python convert_dpr_original_checkpoint_to_pytorch.py \
--type ctx_encoder \
--src ../DPR-main/outputs/2023-04-15/18-54-14/dpr_biencoder.$1 \
--dest ctx_encoder 

cp pytorch/question_encoder/tokenizer_config.json question_encoder/

cp pytorch/ctx_encoder/tokenizer_config.json ctx_encoder/

cp pytorch/question_encoder/vocab.txt question_encoder/

cp pytorch/ctx_encoder/vocab.txt ctx_encoder/

python qa_system.py












