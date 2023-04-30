import sys 
sys.path.append("../utils")
sys.path.append("../DPR-main/")

from f1_score import calculate_f1

import csv, pickle, torch, time, json
from transformers import (
    DPRReader, DPRReaderTokenizer
    )



# ############################################### READER ##############################################################
# facebook/dpr-reader-single-nq-base
def extractive_reader(oracle_json, reader, span_answers):
    
    tokenizer = DPRReaderTokenizer.from_pretrained(reader)
    model = DPRReader.from_pretrained(reader).to('cuda:0')

    total = 0
    with open(span_answers, "w", encoding = "utf-8") as fout:
        
        with open(oracle_json, encoding='utf-8') as f:
            oracles = json.load(f)
            for oracle in oracles:
                q, t, c = oracle['question'], oracle['ctxs'][0]['title'], oracle['ctxs'][0]['text']
                encoded_inputs = tokenizer(q, t, c, return_tensors = "pt", 
                    max_length = 300, padding = 'max_length', truncation = True).to('cuda:0')

                outputs = model(**encoded_inputs)
                predicted_spans = tokenizer.decode_best_spans(encoded_inputs, 
                                                              outputs, 
                                                              max_answer_length = 20, 
                                                              num_spans = 1, 
                                                              num_spans_per_passage = 1)
                    
                fout.write("{}\t{}\n".format(q, predicted_spans[0].text))

                total += 1
                if(total % 10 == 0):
                    print("Extracted spans for {} questions.".format(total)) 


def validate_reader(questions, span_answers):
    
    f1, em = [], []
    qa_dic = {}
    
    with open(questions, "r", encoding = "utf-8") as fin:
         reader = csv.reader(fin, delimiter = "\t")
         for row in reader:
             question = row[0]
             answer = row[1].strip('"["').strip('"]"')
             qa_dic[question] = answer.replace(".", "")
             
    with open(span_answers, "r", encoding = "utf-8") as fin:
       reader = csv.reader(fin, delimiter = "\t")
       for row in reader:
           question = row[0]
           
           answer = row[1].strip().replace(".", "").replace(" %", "%")
           
           macro_f1 = calculate_f1(qa_dic[question], answer)   
           f1.append(macro_f1)
           
           if(macro_f1 == 1):
               em.append(1)
           else:
               em.append(0)

 
    print("em: {:.2f}, f1: {:.2f}".format(sum(em)/len(em), sum(f1)/len(f1)))
    
if __name__ == '__main__':

    questions = "../data/training/sleep-test.csv"

    test_data = '../data/training/oracle/sleep-test.json'
    span_answers = "processed/best_reader_predicted_spans.csv"
    reader = "reader/"
#    reader = 'deepset/bert-base-uncased-squad2'
    
    extractive_reader(test_data, reader, span_answers)
    validate_reader(questions, span_answers)

