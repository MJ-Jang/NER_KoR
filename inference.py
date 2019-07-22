import torch
import torch.optim as optim
import os, sys

from model import BiLSTMCRF
from utils import f1_score, get_args, get_tags_BIE, DataManager_BIE
import copy
import sentencepiece as spm
import json
import numpy as np
import re
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings(action='ignore')

def load_model():
    model_path = './model/BIO_BiLSTMCRF_v1.0_200.pkl'
    tokenizer_path = './data/TWD_one_to_one_raw_sentencepiece.model'

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(tokenizer_path)

    parser = get_args()
    args = parser.parse_args()
    
    ## For loading tag meta data ##
    with open('./data/NER_data_v1.0.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    args.tags = data['categories']
    del(data)

    inference_manager = DataManager_BIE(args, data_type = 'inference')
    tag_map = inference_manager.tag_map    
    
    id2tag = {}
    for k, v in tag_map.items():
        id2tag[v] = k

    model = BiLSTMCRF(
        tag_map=tag_map,
        vocab_size=tokenizer.get_piece_size(),
        embedding_dim=300,
        hidden_dim=512,
        num_layer=1
    )
    model.load_state_dict(torch.load(model_path))
    
    return (model, tokenizer, tag_map, id2tag)


def inference(models, text):
    model, tokenizer, tag_map, id2tag = models

    if type(text) != list and type(text) == str:
        text = [text]

    text = [t.lower() for t in text]
    # copy and basic preprocess #
    text_copy = copy.deepcopy(text)
    text_copy = [re.sub(pattern=r"\s\s+", repl=' ', string=t) for t in text_copy]

    text_tok = [tokenizer.EncodeAsPieces(t) for t in text_copy]
    text_idx = [tokenizer.EncodeAsIds(t) for t in text_copy]

    max_len = max(map(lambda x: len(x), text_idx))
    text_idx = [s + [0]*(max_len - len(s)) for s in text_idx]
    text_input = torch.tensor(text_idx, dtype=torch.long)

    _, path, probs = model(text_input)
    print(path)
    
    def NER_decode(path, text, text_tok):

        detected_tag = [id2tag[tag] for tag in path[:len(text_tok)] if tag not in [tag_map['B-O'], tag_map['I-O'], tag_map['E-O']]]
        detected_tag = np.unique([s.split('-')[1] for s in detected_tag]).tolist()
        print(detected_tag)

        result = []
    
        for i, tag in enumerate(detected_tag):
            positions = get_tags_BIE(path, tag, tag_map)
            #print(positions)
 
            for idx, (s,e) in enumerate(positions):
                entity = tokenizer.DecodePieces(text_tok[s:e+1])
                prob = np.mean(probs[s:e+1])
                #print(entity)

                if len(entity) == 0:
                    continue
                else:
                    try:
                        entity_pos = re.search(pattern=entity, string=text).span()
                        #print(entity_pos)
                    
                        tmp_result = OrderedDict()
                        tmp_result['entity_type'] = re.sub(pattern = '[<>]+', repl = '', string=tag)    
                        tmp_result['entity_value'] = entity               
                        tmp_result['start_pos'] = entity_pos[0]
                        tmp_result['end_pos'] = entity_pos[1]
                        tmp_result['score'] = round(prob, 2)
                        tmp_result['isPredict'] = True
                    
                        #text_copy = re.sub(entity, '#'*len(entity), text_copy)
                        text = text.replace(entity, '#'*len(entity), 1)
                        #print(text)
                        result.append(tmp_result)
                    except:
                        continue
        return result
        
    result = [NER_decode(path[k], text_copy[k], text_tok[k]) for k in range(len(path))]
                
    return result

if __name__ == '__main__':
    models = load_model()

    text = ['아이폰6 구매하려고 합니다.', '갤럭시S9 64g로 기기변경 하고싶어요', '슈퍼쏜 슈퍼콘 슈퍼슈퍼슈퍼쏜']
    result = inference(models, text)

    for i, r in enumerate(result):
        print('Input text: {}'.format(text[i]))
        print('Results:')
        for entities in r:
            print(entities)
        print('-'*30)



    
