# -*- coding:utf-8 -*-

import argparse
import copy
import numpy as np
import torch.nn.functional as F
import sentencepiece as spm
import json
import re
from easydict import EasyDict


class JsonConfig:
    def __init__(self, file_path):
        self.values = EasyDict()
        if file_path:
            self.file_path = file_path
            self.reload()

    def reload(self):
        self.clear()
        if self.file_path:
            with open(self.file_path, 'r') as f:
                self.values.update(json.load(f, encoding='utf-8'))

    def clear(self):
        self.values.clear()

    def update(self, in_dict):
        for (k1, v1) in in_dict.items():
            if isinstance(v1, dict):
                for (k2, v2) in v1.items():
                    if isinstance(v2, dict):
                        for (k3, v3) in v2.items():
                            self.values[k1][k2][k3] = v3
                    else:
                        self.values[k1][k2] = v2
            else:
                self.values[k1] = v1

    def export(self, save_file_name):
        if save_file_name:
            with open(save_file_name, 'w') as f:
                json.dump(dict(self.values), f)



class DataManager():
    def __init__(self, args, data_type='train'):
        self.input_size = 0
        self.batch_size = args.batch_size
        self.max_length = args.max_length
        self.data_type = data_type
        self.data = []
        self.batch_data = []

        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(args.tokenizer_path)
        self.tagger = NER_tagger(args.tokenizer_path)

        self.tags = args.tags
        self.tag_map = args.tag_map

        if data_type == "train":
            self.data_path = args.input_path
            self.load_data()
            self.prepare_batch()

        elif data_type == "dev":
            self.data_path = args.input_path
            self.load_data()
            self.prepare_batch()

        elif data_type == "inference":
            pass

    def load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        if self.data_type == 'train':
            dataset = loaded['train']

        elif self.data_type == 'dev':
            dataset = loaded['dev']

        for i, dat in enumerate(dataset):
            sent, tag = self.tagger.process(dat, 'BIO')
            sent = [self.tokenizer.piece_to_id(p) for p in sent]
            tag = [self.tag_map[t] for t in tag]

            self.data.append([sent, tag])

        self.input_size = self.tokenizer.get_piece_size()
        print("{} data: {}".format(self.data_type, len(self.data)))
        print("vocab size: {}".format(self.input_size))
        print("unique tag: {}".format(len(self.tag_map.values())))

    def prepare_batch(self):
        '''
            prepare data for batch
        '''
        index = 0
        while True:
            if index + self.batch_size >= len(self.data):
                pad_data = self.pad_data(self.data[-self.batch_size:])
                self.batch_data.append(pad_data)
                break
            else:
                pad_data = self.pad_data(self.data[index:index + self.batch_size])
                index += self.batch_size
                self.batch_data.append(pad_data)

    def pad_data(self, data):
        c_data = copy.deepcopy(data)
        max_length = max([len(i[0]) for i in c_data])
        for i in c_data:
            i.append(len(i[0]))
            i[0] = i[0] + (max_length - len(i[0])) * [0]
            i[1] = i[1] + (max_length - len(i[1])) * [0]


        return c_data

    def iteration(self):
        idx = 0
        while True:
            yield self.batch_data[idx]
            idx += 1
            if idx > len(self.batch_data) - 1:
                idx = 0


    def get_batch(self):
        for data in self.batch_data:
            yield data


class NER_tagger:
    def __init__(self, tok_path):
        '''
        tok_path: sentence piece tokenizer model path
        output_type: output type of sentence, token or index
        '''
        self.tok = spm.SentencePieceProcessor()
        self.tok.Load(tok_path)

    def _extract_entity(self, sentence):
        entity_dict = []
        
        tags = re.findall(pattern='<[a-zA-Z]+>[^<>]+<[/a-zA-Z]+>', string=sentence)
        for tag in tags:
            tag = re.sub(pattern='</[a-zA-Z]+>', repl='', string=tag)
            key, value = tag.split('>')
            entity_dict.append([value, key + '>'])
        return entity_dict
    
    def process(self, sentence, tag_type):
        '''
            sentence: input sentence with NER tagged (ex: <Model>아이폰X</Model> 구매하려구요)
            tag_type: type of tag ('XO','BIO','raw')
            '''
        if tag_type not in ['XO', 'BIO', 'raw']:
            raise ValueError('tag type should be "XO","BIO", or "raw"')
        
        original_sentence = re.sub(pattern='<[a-zA-Z/]+>', repl='', string=sentence)

        entity_dict = self._extract_entity(sentence)

        sent_token = self.tok.EncodeAsPieces(original_sentence)

        tags = ['O'] * len(sent_token)

        s_pos = 0
        for value, tag in entity_dict:
            value_token = self.tok.EncodeAsPieces(value)

            v_idx = 0

            if tag_type == 'BIO':
                pre_fix = 'B-'
            else:
                pre_fix = ''

            for s_idx, s in enumerate(sent_token[s_pos:]):
                if v_idx == 0 and len(value_token) > 1:
                    next_id = s_pos + s_idx + 1

                    if next_id == len(sent_token):
                        next_id -= 1
                    if sent_token[next_id] != value_token[v_idx + 1]:
                        continue

                if s == value_token[v_idx]:
                    if tag_type == 'XO' and v_idx != 0:
                        tags[s_pos + s_idx] = 'X'
                    else:
                        tags[s_pos + s_idx] = pre_fix + tag
                    v_idx += 1

                    if tag_type == 'BIO':
                        pre_fix = 'I-'

                if v_idx == len(value_token):
                    s_pos = s_pos + v_idx
                    break
                    
        return (sent_token, tags)


def f1_score(tar_path, pre_path, tag, tag_map, length):
    origin = 0.
    found = 0.
    right = 0.
    for idx, fetch in enumerate(zip(tar_path, pre_path)):
        tar, pre = fetch
        tar_tags = get_tags_BIE(tar[:length[idx]], tag, tag_map)
        pre_tags = get_tags_BIE(pre[:length[idx]], tag, tag_map)

        origin += len(tar_tags)
        found += len(pre_tags)

        for p_tag in pre_tags:
            if p_tag in tar_tags:
                right += 1

    recall = 0. if origin == 0 else (right / origin)
    precision = 0. if found == 0 else (right / found)
    f1 = 0. if recall+precision == 0 else (2*precision*recall)/(precision + recall)
    print("\t{}\trecall {:.2f}\tprecision {:.2f}\tf1 {:.2f}".format(tag, recall, precision, f1))
    return recall, precision, f1


def get_tags(path, tag, tag_map):
    tag_idx = tag_map.get(tag)
    positions = [i for i, x in enumerate(path) if x == tag_idx]

    positions = sorted(positions)
    gaps = [[s, e] for s, e in zip(positions, positions[1:]) if s + 1 < e]
    edges = iter(positions[:1] + sum(gaps, []) + positions[-1:])

    tags = [list(t) for t in list(zip(edges, edges))]
    return tags


def get_tags_BIE(path, tag, tag_map):
    begin_tag = tag_map.get("B-" + tag)
    mid_tag = tag_map.get("I-" + tag)
    end_tag = tag_map.get("E-" + tag)
    #single_tag = tag_map.get("S")
    o_list = [tag_map.get("B-O"), tag_map.get("I-O"), tag_map.get("E-O")]
    begin = -1
    end = 0
    tags = []
    last_tag = 0
    for index, tag in enumerate(path):
        if tag == begin_tag and index == 0:
            begin = 0
        elif tag == begin_tag:
            begin = index
        elif tag == end_tag and last_tag in [mid_tag, begin_tag] and begin > -1:
            end = index
            tags.append([begin, end])
        elif tag in o_list:
            if last_tag == begin_tag:
                tags.append([begin, begin])
            else:
                begin = -1
        last_tag = tag
    return tags

###


artifact_path = 'C:\\Users\\MJ_Jang\\Desktop\\workspace\\NER_all_in_one/'


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default="./data/NER_data_v1.0.json")
    parser.add_argument('--input_size', type=int, default=8000)
    parser.add_argument('--vocab', type=dict, default={})
    parser.add_argument('--tag_map', type=dict, default={})
    parser.add_argument('--tags', type=list, default=['<Model>', 'O'])
    parser.add_argument('--tokenizer_path', type=str,
                        default=artifact_path + '/data/TWD_one_to_one_raw_sentencepiece.model')

    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--num_layer', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--cuda', type=bool, default=True)

    # parser.add_argument('--is_cuda', type=bool, default=False)

    return parser