import torch
import torch.optim as optim
import os, sys

from model import BiLSTMCRF
from utils import f1_score, get_tags_BIO, DataManager, JsonConfig
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
    model_path = './BIO_BiLSTMCRF_Focal_v1.0_100.pkl'
    tokenizer_path = './data/tokenizer.model'

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(tokenizer_path)

    params = JsonConfig('./params_BiLSTM.json')
    args = params.values
    args.cuda = False

    # load tag_map
    with open(args.tag_map_path, 'r', encoding='utf-8') as file:
        tag_map = json.load(file)

    args.tag_map = tag_map
    tags = [tag[2:] for tag in tag_map.keys() if '<' in tag]
    tags = list(np.unique(tags))
    args.tags = tags   
    
    id2tag = {}
    for k, v in tag_map.items():
        id2tag[v] = k

    model =  BiLSTMCRF(
        tag_map=tag_map,
        batch_size=args.batch_size,
        vocab_size=tokenizer.get_piece_size(),
        dropout=args.dropout,
        embedding_dim=args.embedding_size,
        hidden_dim=args.hidden_size,
        num_layer=args.num_layer,
        use_cuda=args.cuda
    )
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    return (model, tokenizer, tag_map, id2tag)

def inference(models, text):
    model, tokenizer, tag_map, id2tag = models
    model.eval()

    pad_token = tokenizer.piece_to_id('<pad>')
    print(pad_token)

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
    print(text_idx)
    text_input = torch.tensor(text_idx, dtype=torch.long)

    _, path, probs = model(text_input)
    
    def NER_decode(path, text, text_tok):
        print(path[:len(text_tok)])
        detected_tag = [id2tag[tag] for tag in path[:len(text_tok)] if 'B-' in id2tag[tag]]
        detected_tag = np.unique([s.split('-')[1] for s in detected_tag]).tolist()

        result = []
    
        for i, tag in enumerate(detected_tag):
            positions = get_tags_BIO(path, tag, tag_map)
            print(positions)
 
            for idx, (s,e) in enumerate(positions):
                entity = tokenizer.DecodePieces(text_tok[s:e+1])
                prob = np.mean(probs[s:e+1])
                print(entity)

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
    tokenizer = models[1]
    tokenizer.piece_to_id('<pad>')

    text = ['미국 피플지',
            'LG전자(066570, www.lge.co.kr)는 10일 LS엠트로는 공조시스템 사업부문 인수 계약을 체결했다.', 
            '우면산, 관악산, 삼성산 : 토 14:00 ~ 토 22:00']
    result = inference(models, text[1])

    for i, r in enumerate(result):
        print('Input text: {}'.format(text[i]))
        print('Results:')
        for entities in r:
            print(entities)
        print('-'*30)