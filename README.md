data: 

data_aug folder - run aug.py to produce training data 
data/training/sleep-train_aug.json and 
sleep-train_aug2.json include sleep-train.jsom and sleep-train_aug.json

Adjust DPR main
under conf -  
1. datasets folder:
encoder_train_default.yaml -> change the path of dataset here

2 under train folder:
biencoder_default.ymal -> set batch size and learning rate
extraive_reader_defualt.yamal -> set batch size, no change to learning rate

biencoder_train_cfg.ymal -> trani_dataset, deve_dataset, output_dir, checkpoint_file+nam e, n_gpu
extractive_readre_train_cgf.ymal -> train files, dev_files, output_dir, checkpoint_file_name, n_gpu



run 
train_dense_encoder.py
trani_extracive_reader.py 

save in the desinated outputs folder specify in ymal.file 

convert under model
convert using 
using convert_reader.sh reader -may change path
convert.sh - retrivel - may change path

test 
retrival qa_system.py -
reader_test.py - because retrival have no end to end,so write a new test code.





Citation to the original paper

Link to the original paper’s repo (if applicable)

Dependencies 
to do 

Data download instruction


Preprocessing code
to do 

Training code + command 
to do 

Evaluation code + command
to do 

Pretrained model 
to do


Table of results (no need to include additional experiments, but main reproducibility result should be included)



