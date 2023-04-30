#!pip install transformers sentencepiece tqdm torch nltk

from tqdm import tqdm
import torch
from nltk.tokenize import sent_tokenize

import nltk
nltk.download('punkt')


#!pip install sacremoses
from transformers import *

torch.manual_seed(1720)

model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")

model_name = "tuner007/pegasus_paraphrase"
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)

model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def get_response(input_text,num_return_sequences,num_beams,temp=2):
    batch = tokenizer([input_text],truncation=True,padding='longest', max_length=60, return_tensors='pt').to(torch_device)
    translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=2)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text


import json 
sleep_train = json.load(open('../data/training/sleep-train.json'))

newdata = []
for example in tqdm(sleep_train):
    q = example['question']
    q_para = get_response(q, 1, 10)[0]
    example['question'] = q_para
    a = example['answers']
    pc = example['positive_ctxs']
    for pci in pc:
        text = pci['text']
        sents = list(sent_tokenize(text))
#        print('question: ', q)
#        print()
#        print('answer:  ', a)
#        print()
#        print()
        text_para = ''
        for i, sent in enumerate(sents):
            sent_para = get_response(sent, 1, 10)[0]
            text_para += ' ' + sent_para
#            print('original: ', sent)
#            print('new para: ', i, sent_para)
#            print()
        pci['text'] = text_para

    pc = example['negative_ctxs']
    for pci in pc:
        text = pci['text']
        sents = list(sent_tokenize(text))
#        print('question: ', q)
#        print()
#        print('answer:  ', a)
#        print()
#        print()
        text_para = ''
        for i, sent in enumerate(sents):
            sent_para = get_response(sent, 1, 10)[0]
            text_para += ' ' + sent_para
#            print('original: ', sent)
#            print('new para: ', i, sent_para)
#            print()
        pci['text'] = text_para

    newdata.append(example)
#    break

json.dump(newdata, open('../data/training/sleep-train_aug2.json','w'), indent=2)