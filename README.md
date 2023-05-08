

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


## Data Preprocessing and Download

The original SleepQA dataset has been pre-processed in DPR required format by the authors, and stored in data/training within three files: sleep-train.json, sleep-dev.json, sleep-test.csv. 

The new augmented dataset SleepQA3x created in this study is via data_aug/aug.py, which automatically reads the data/training/sleep-train.json. To install dependencies, run command in data_aug/req.txt. The new train dataset is stored as data/training/sleep-train_aug2.json (aug2 means the sleep-train.json was paraphrased twice, and orginal sleep-train.json has been manually incorporated into sleep-train_aug2.json, resulting an augmented train dataset 3 times the original SleepQA).

To train model with this augumented dataset, simply change the DRP setting with DPR-main/conf/datasets/encoder_train_default.yaml: 

```
sleep_train:
    _target_: dpr.data.biencoder_data.JsonQADataset
    file: "../../../../data/training/sleep-train_aug2.json"
```



## Training:

There are two models to train:
  1. Retriever, which retrieves relevant passages from a corpus of sentences for a given question;
  2. Reader, which extracts answers from a passage for a given question.


### Train retriever:

Configurate the .yaml files properly before training. Retriever uses 4 config files in DPR-main: 
- Main configuration: conf/biencoder_train_cfg.yaml <br />
It has been setup using hf_PubMedBERT as encoder in name of encoder parameter. Set the number of gpu for n_gpu if GPU can be used for training, otherwise, n_gpu should be set to 0, and no_cuda set to True.

- Dataset configuration: conf/datasets/encoder_train_default.yaml <br />
Change the path of dataset accordingly if needed.

- Encoder configuration: conf/encoder folder <br />
It stores all the encoder models selected by the authors, including the hf_PubMedBERT.yaml that was used in this study. 

- Train configuration: conf/train/biencoder_default.yaml <br />
Set the hyperparameter for retreiver. The batch_size is set to 3 due to memory limits (a single GPU RTX3070 Ti with 8G memory was used in this reproduction study). Larger batch size is recommended (a batch size of 16 was used by the authors) since contrastive learning utilized by DPR can be improved if more negative pairs can be incorporated in a batch.
    
To train retriever, run DPR-main/train_dense_encoder.py. No hyperparameters should be required if the configurations are set properly. Note that the outputs will be written into DPR-main/outputs/yyyy-mm-dd where the date is the date when the training kicks off. 

### Train reader:

Similar to retreiver, configuarate the yaml files properly before training reader:

- Main configuration: conf/extractive_reader_train_cfg.yaml <br />
It has been set using hf_BioASQ as encoder in name of encoder parameter. Set the number of gpu for n_gpu parameter if GPU can be used for training. Otherwise, n_gpu should be set to 0, and no_cuda set to True.
  
- encoder configuration:  <br />
conf/encoder/hf_BioASQ.yaml is the one used in this study, no need to change the configuration.
  
- train configuration: conf/train/extractive_reader_default.yaml  <br />
Set the hyperparameter for reader. The batch_size is set to 3 due to memory limits. A larger batch size may help improving accuracy, but unlike retriever training which requires contrastive loss, reader model trains each example independently without using in-batch pairs. As a result, increasing the batch size only helps to reduce gradient noise without providing other additional benefits. Standard size that generally works well is 16 (used by the authors of SleepQA) or 32. One can change accordingly if GPU memory is sufficient.

After setting the configs, to train reader, run train_extractive_reader.py. The outputs will be saved in DPR-main/outputs/yyyy-mm-dd. 


## Evaluation:

To evaluate the trained models, follow two steps: 1. Convert models from DPR models to Pytorch models; 2. Test Model: Run qa_system.py (for both retriever and reader end-to-end evaluation), or reader_test.py for reader evaluation only.  

### Model Conversion: 
The DPR model checkpoints for retriever and reader are saved in DPR-main/outputs/yyyy-mm-dd. One needs to run convert_dpr_original_checkpoint_to_pytorch.py to convert to Pytorch models.

To make this step easy, two scripts were created in this study: convert.sh to convert retriever checkpoints and convert_reader.sh to convert reader checkpoints. 

convert_retriever.sh: Once a model is trained, one need to change the --src to the proper trained model path in /DPR-main/outputs/ folder. $1 is used to choose which checkpoint to use. Note that both the question_encoder and ctx_encoder use the same input checkpoint. Then, the rest of the codes do not need to be changed. 

convert_reader.sh: one need to change --src to the propoer DPR-main/outputs/ checkpoint name. The rest of the codes do not need to be changed. 


### Model evaluation

reader_test.py: it takes the test questions from data/training/sleep-test.csv as input, test data in data/training/oracle/sleep-test.json for ground truth labels, uses the reader model saved in models/reader/ folder after 'model conversion' step, and save predicted answers in models/processed/best_reader_predicted_spans.csv for validation. The program will print out the validation results with EM and F1 score.


qa_system.py: to use the converted models, one need to set the ctx encoder: ctx_encoder='pytorch/ctx_encoder', question encoder: 'pytorch/question_encoder', and reader: reader='pytorch/reader'. The other codes do not need to change. 

qa_system.py is the entire question answering pipeline. It first, for each question, retrieves the top 1 passage from the entire corpus, and then use the reader to read on this passage and extract a single span as answer. Specifically, it takes the text_corpus and questions as input, so that the ctx_encoder and question_encoder can encode them and conduct embedding-based search for each question. The produced embedded corpus will be saved in processed/sleep-corpus_e29_aug, the retrieved passages will be saved in processed/sleep_test_e29_aut.csv. Then, the reader saved in pytorch/reader will use both the retrieved passage and the question altogether to extract span answers that are saved in 'processed/pipeline1_label_1.250.aug.csv. 

Note that, qa_system.py will output retrieval-step Top-1 document hits (Recall@1)  for retriever evaluation, and also end-to-end EM and F1 for the entire pipeline. 


## Models: 

For reproduction study, the best fine-tuned models obtained from this reproduction study are stored in the models/ folder: models/question_encoder/, models/ctx_encoder/, models/reader/. Those models are git lfs files, which can be directly used for reader_test.py and qa_system.py evaluation scripts. No conversion is needed. 

For SleepQA3x augumentation, this study uses Google Pegasus paraphrasing model 'tuner007/pegasus_paraphrase' in data_aug/aug.py

For baseline retriever, this study used DPR BERT-based retriever trained on Natural Questions (NQ) dataset. <br />
Set ctx_encoder='facebook/dpr-ctx_encoder-single-nq-base' and question_encoder= ''facebook/dpr-question_encoder-single-nq-base' in qa_system.py


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

