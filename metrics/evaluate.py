import json
from metrics.f1_score import f1_score
from metrics.exact_match_score import exact_match_score
from dataloader import *

def evaluate(predictions, mode):
    list_sample = []

    if mode == 'dev':
        path = './DataNew/dev_ViQuAD.json'
    elif mode == 'test':
        path = './DataNew/test_ViQuAD.json'
    else:
        raise Exception("Only dev and test dataset available")
        
    f1 = exact_match = total = 0
    
    list_sample = InputSample(path=path, max_char_len=10, max_seq_length=250, stride=10).get_sample()
    for i, sample in enumerate(list_sample):

        context = sample['context']
        question = sample['question']
        sentence = ['cls'] + question + ['sep'] + context

        labels = sample['label_idx']

        f1_idx = [0]
        extract_match_idx = [0]
        for lb in labels:

            start = lb[1]
            end = lb[2]
            ground_truth = " ".join(sentence[start:end+1])
            
            start_pre = int(predictions[i][1])
            end_pre = int(predictions[i][2])
            label_prediction = " ".join(sentence[start_pre:end_pre+1])
            f1_idx.append(f1_score(label_prediction, ground_truth))
            extract_match_idx.append(exact_match_score(label_prediction, ground_truth))

            # print(ground_truth)
            # print(label_prediction)


        f1 += max(f1_idx)
        exact_match += max(extract_match_idx)
        total += 1
        
    if total == 0:
        f1 = 0
        exact_match = 0
    else:
        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total
    
    return exact_match, f1