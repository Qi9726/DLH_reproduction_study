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



# A Reproduction Study for SleepQA: Dataset on Sleep for Extractive Question Answering

This study is to replicate the paper: Iva Bojic, Qi Chwen Ong, Megh Thakkar, Esha Kamran, Irving Yu Le Shua, Rei Ern Jaime Pang, Jessica Chen, Vaaruni Nayak, Shafiq Joty, Josip Car. SleepQA: A Health Coaching Dataset on Sleep for Extractive Question Answering. Proceedings of Machine Learning for Health (ML4H) 2022 Workshop.

The codebase of the reproduction study is built upon SleepQA repo https://github.com/IvaBojic/SleepQA and Dense Passage Retrieval (DPR) repo https://github.com/facebookresearch/DPR, which was used by the authors of SleepQA to build their question answering (QA) pipeline. 

New files created in this study are as below:
- SleepQA3x data augumentation: data_aug/aug.py, data_aug/req.txt
- SleepQA3x train dataset: data/training/sleep-train_aug.json, data/training/sleep-train_aug2.json
- Reader only evaluation: models/reader_test.py
- Dependencies: requirements.txt
- Convert model checkpoints to pytorch: models/convert_reader.sh, models/convert.sh
- Best model checkpoints: models/question_encoder/ folder, models/ctx_encoder/ folder, models/reader/ folder



## Dependencies 
Required packages are listed in requirements.txt


## Data Preprocessing and download

Original SleepQA dataset have been pre-processed in DPR required format by the authors, and stored in data/training within three files: sleep-train.json, sleep-dev.json, sleep-test.csv. 

The new augmented dataset SleepQA3x created in this study is via data_aug/aug.py, which automatically reads the data/training/sleep-train.json. To install dependencies, run command in data_aug/req.txt. The new train dataset is stored as data/training/sleep-train_aug2.json (aug2 means the sleep-train.json was paraphrased twice, and orginal sleep-train.json has been manually incorporated into sleep-train_aug2.json, resulting an augmented train dataset 3 times the original SleepQA).

To train model with this augumented dataset, simply change the DRP setting with DPR-main/conf/datasets/encoder_train_default.yaml: 
```
sleep_train:
    _target_: dpr.data.biencoder_data.JsonQADataset
    file: "../../../../data/training/**sleep-train_aug2.json**"
```



## Training:

There are two models to train:
  1. Passage retriever, which retrieves relevant passages from a corpus of sentences for a given question;
  2. passage reader, which extracts answers from a passage for a given question.

For retriever, pre-train the sentence encoder using hf_PubMedBERT. This BERT-base model is first downloaded from huggingface, and then fine-tuned on SleepQA dataset with Facebook's Dense Passage Retriever tool (DPR-main) folder. 

For passage reader training, the same files are used as passage retrieval training, namely, sleep-train.json, sleep-dev.json.


### Train retriever:

Configurate the yaml files properly before training. Retriever uses 4 config files in DPR-main: 
    - Main configuration: conf/biencoder_train_cfg.yaml
    It has been setup using hf_PubMedBERT as encoder in name of encoder parameter. Set the number of gpu for n_gpu if GPU can be used for training. Otherwise, n_gpu should be set to 0, and no_cuda set to True.

    - Dataset configuration: conf/datasets/encoder_train_default.yaml. 
     Change the path of dataset accordingly if needed. To use new augumented dataset, the SleepQA3x, change it to following:
     
sleep_train:
  _target_: dpr.data.biencoder_data.JsonQADataset
  file: "../../../../data/training/**sleep-train_aug2.json**"

- encoder configuration: conf/encoder. 
 It stores all the encoder models selected by the authors, including the hf_PubMedBERT.yaml that used in this study. 

- train configuration: conf/train/biencoder_default.yaml
 Set the hyperparameter for retreiver here: The batch_size is set to 3 due to memory limits (a single GPU RTX3070 Ti with 8G memory was used in this reproduction study). Larger batch size is recommended (a batch size of 16 was used by the authors) since contrastive learning utilized by DPR can be improved if more negative pairs can be incorporated in a batch.
    
To train retriever, run DPR-main/train_dense_encoder.py. No hyperparameters should be required if the configurations are set properly. Note that the outputs will be written into DPR-main/outputs/yyyy-mm-dd where the date is the date that the training is kicked off. 

### Train reader:

Similar to retreiver, configuarate the yaml files properly before training reader:

- Main configuration: conf/extractive_reader_train_cfg.yaml
  It has been set using hf_BioASQ as encoder in name of encoder parameter. Set the number of gpu for n_gpu parameter if GPU can be used for training. Otherwise, n_gpu should be set to 0, and no_cuda set to True.
  
- encoder configuration: 
  conf/encoder/hf_BioASQ.yaml is the one used in this study, no need to change the configuration
  
- train configuration: conf/train/extractive_reader_default.yaml 
   Set the hyperparameter for reader here: The batch_size is set to 3 due to memory limits. A larger batch size may help training accuracy, but unlike retriever training which requires contrastive loss, reader training does not construct in-batch pairs, and each example is independen, thus increasing the batch size to larger size will only help reducing gradient noise, but not introducing additional benefit on top of it. Standard size that generally works well is 16 (used by SleepQA authors) or 32. Can change accordingly if GPU memory is enough.

After setting the configs, to train reader, run train_extractive_reader.py. The outputs will be saved in DPR-main/outputs/yyyy-mm-dd. 


## Evaluation code:

To evaluate the trained models, follow two steps: 1. Convert models from Pytorch models to DPR models; 2. Test Model: Run qa_system.py (for both retriever and reader end-to-end evaluation by SleepQA authors), or reader_test.py for reader evaluation only created in this study. 

### Model Conversion
The DPR model checkpoints for retriever and reader are saved in DPR-main/outputs/yyyy-mm-dd. However, one needs to run convert_dpr_original_checkpoint_to_pytorch.py to convert to Pytorch models.

To make this step easy, two scripts were created in this study: convert.sh to convert retriever checkpoints and convert_reader.sh to convert reader checkpoints to Pytorch models. 

convert.sh: Once a model is trained, one need to change the --src to the proper trained model path in /DPR-main/outputs/ folder. $1 is used to choose which checkpoint to use. Note that both the question_encoder and ctx_encoder use the same input checkpoint. Then, the rest of the codes do not need to be changed. Note that `python qa_system.py` in line 23 at the end is the end-to-end retriever-reader testing code, which can be commented out if one just want to convert models, but not testing them. 

Similarly for convert_reader.sh, one need to change --src to the propoer DPR-main/outputs/ checkpoint name. The rest of the codes do not need to be changed. ``python reader_test.py`` is the reader test file, which can also be commented out If one just want to convert model without testing.


### Model Test

Once the model conversion is done, the models necessary for this test step should have been placed in the proper folders, and the two scripts should be able to just call the models and run the test without problem.

For reader_test.py, it takes the test questions from data/training/sleep-test.csv as input, test data data/training/oracle/sleep-test.json for ground truth labels, uses the reader model saved in models/reader/ folder after 'model conversion' step, and save predicted answers in models/processed/best_reader_predicted_spans.csv for validation. The program will print out the validation results with EM and F1 score.


For qa_system.py, to use the converted models, one need to change the ctx_encoder and question_encoder to ctx_encoder='pytorch/ctx_encoder' and question_encoder to 'pytorch/question_encoder', and reader='pytorch/reader'. The other codes do not need to change. 

qa_system.py is the entire question answering pipeline. It first, for each question, retrieve the top 1 passage from the entire corpus, and then use the reader to read on this passage and extract a single span as answer. Specifically, it takes the text_corpus and questions as input, so that the ctx_encoder and question_encoder can encode them and conduct embedding-based search for each question. The produced embedded corpus will be saved in processed/sleep-corpus_e29_aug, the retrieved passages will be saved in processed/sleep_test_e29_aut.csv. Then, the reader saved in pytorch/reader will use both the retrieved passage and the question altogether to extract span answers that are saved in 'processed/pipeline1_label_1.250.aug.csv. 

Note that, qa_system.py will output retrieval-step Top-1 document hits (Recall@1)  for retriever evaluation, and also end-to-end EM and F1 for the entire pipeline. 

### Test with a non-fine-tuned text encoder
For ablation, this study used DPR BERT-based retriever trained on Natural Questions (NQ) dataset as baseline retreival model. Set ctx_encoder='facebook/dpr-ctx_encoder-single-nq-base' and question_encoder= ''facebook/dpr-question_encoder-single-nq-base' in qa_system.py to run these   two models.

## models: 

The best fine-tuned models obtained from this reproduction study are stored in the models/ folder, namely, question_encoder/, ctx_encoder/, and reader/ 

Those models are git lfs files, which can be directly used for reader_test.py and qa_system.py evaluation scripts. No conversion is needed. 

For SleepQA3x, Google Pegasus paraphrasing model 'tuner007/pegasus_paraphrase' is set in data_aug/aug.py

## Data Analysis

Comparison of average number of words in passage and question in train sets of original SleepQA and augumented SleepQA3x.

| Dataset   | Passage  | Question |
| --------- |----------- | ------ |
| SleepQA   |    120.5   |  9.9   |
| SleepQA3x |     97.7.  |  9.1   |


## Reproduction and Ablation Results 

### Retriver 

| Results   | Model  | recall@1 | 
| --------- |----------- | ------ |
| Original Paper |    Fine-tuned BERT (SleepQA)  |  0.35  | 
| Original Paper  |     Fine-tuned PubMedBERT (SleepQA) |  0.42   | 
| --------- |----------- | ------ |
| Reproduction Study|  Fine-tuned PubMedBERT (SleepQA)  |  0.35   |
| Ablation Study |  Fine-tuned PubMedBERT (SleepQA3x)  |  0.35  | 
| Ablation Study |  DPR BERT (Natual Question)   |  0.18   |

This reproduction study chose PubMedBERT to fine tune from scratch as it is the best-performing fine-tuned retriever of the SleepQA paper. Despite using the same DPR framework and codebase as the original paper did, this study did not yield comparable recall@1 score (0.35 vs 0.42), although it matches the recall@1 score of the fine-tuned general-domain BERT base model reported in the original paper. A small batch size of 3 was used due to the memory limitations, which may have resulted in failure in replication, as the DPR retriever's in-batch negative technique relies on having enough negative pairs per question. With a batch size of 3, there were only 2 negative pairs per question, compared to 15 for a batch size of 16 used by the authors.

In an attempt to address the limitations of a small batch size and improve retrieval performance, this study utilized the augmented SleepQA3x dataset to fine-tune PubMedBERT. However, the score remained at 0.35. This may be because the training objective of the DPR retriever is to create dense embeddings that can capture sentence semantics, and SleepQA3x created via paraphrasing that only preserves the original meaning does not add diversity to the semantic information in the original SleepQA dataset, thus not effectively training a better retriever.

To access the value of SleepQA dataset, this study used DPR BERT pretrained on Natural Questions (NQ) dataset as a baseline, which resulted in a recall@1 score of only 0.18. This upholds the importance of utilizing domain-specific datasets for domain-specific tasks, specifically, the value of SleepQA dataset to sleep health domain, despite their significantly smaller size (4,000) when compared to open-domain datasets (almost 60,000).



#### Reader 

| Results   | Model  | EM (oracle) | F1 (oracle)| 
| --------- |----------- | ------ | ------ |
| Original Paper |    BERT SQuAD2  |  0.50  | 0.64  | 
| Original Paper  |     Fine-tuned BioBERT BioASQ |  0.61   | 0.73   | 
| --------- |----------- | ------ |
| Reproduction Study |  Fine-tuned BioBERT BioASQ  |   0.61  | 0.72  |


To verify reader performance, this study experimented with BioBERT BioASQ model, which was identified as the highest-performing reader in the
original paper. In contrast to the retriever, this study was able to achieve the similar level of performance as the SleepQA paper, despite using a small batch size of 3. The reason can be that the reader model does not involve contrastive learning, making the batch size less impactful on training performance. This finding confirms both the reader performance reported in the original paper and the fact that fine-tuning with SleepQA can generate better outcomes compared to the baseline model (BERT SQuAD2).

