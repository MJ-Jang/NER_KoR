import json
import numpy as np
import re

with open('./data/NEtaggedCorpus_train.json', 'r', encoding='utf-8') as file:
    data = json.load(file)


result = []

for s in data['sentence']:

    text = s['text']
    NEs = s['NE']

    pos = 0
    for NE in NEs:
        entity = NE['text']
        repl = '<' + NE['type'] + '>' + entity + '</' + NE['type'] + '>'

        text_left = text[:pos]
        text_right = text[pos:]

        text_right = text_right.replace(entity, repl, 1)
        text = text_left + text_right

        s, e = re.search(pattern='</' + NE['type'] + '>', string=text_right).span()
        pos = e + len(text_left)

    result.append(text)

np.random.seed(1234)
ids = np.random.permutation(len(result))

id_tr = ids[:int(len(result) * 0.9)]
id_dev = ids[int(len(result) * 0.9):]

result_tr = [result[i] for i in id_tr]
result_dev = [result[i] for i in id_dev]

NER_data = {}
NER_data['train'] = result_tr
NER_data['dev'] = result_dev

with open('./data/NEtagged_data.json', 'w', encoding='utf-8') as file:
    json.dump(NER_data, file, ensure_ascii=False)

