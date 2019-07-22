# -*- coding:utf-8 -*-

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

START_TAG = "START"
STOP_TAG = "STOP"

def log_sum_exp(vec):
    max_score = torch.max(vec, 0)[0].unsqueeze(0)
    max_score_broadcast = max_score.expand(vec.size(1), vec.size(1))
    result = max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), 0)).unsqueeze(0)
    return result.squeeze(1)

class BiLSTMCRF(nn.Module):

    def __init__(
            self, 
            tag_map={},
            batch_size=20,
            vocab_size=20,
            hidden_dim=128,
            dropout=1.0,
            embedding_dim=100,
            num_layer=1,
            use_cuda = False
        ):
        super(BiLSTMCRF, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.num_layer = num_layer
        self.dropout = dropout
        self.use_cuda = use_cuda
        
        self.tag_size = len(tag_map)
        self.tag_map = tag_map
        
        self.transitions = nn.Parameter(
            torch.randn(self.tag_size, self.tag_size)
        )
        self.transitions.data[:, self.tag_map[START_TAG]] = -1000.
        self.transitions.data[self.tag_map[STOP_TAG], :] = -1000.

        self.word_embeddings = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                        num_layers=self.num_layer, bidirectional=True, batch_first=True, dropout=self.dropout)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_cuda:
            hidden = [init.uniform_(torch.randn(2*self.num_layer, self.batch_size, self.hidden_dim // 2), -1, 1).cuda()
                      for _ in range(2)]

        else:
            hidden = [init.uniform_(torch.randn(2*self.num_layer, self.batch_size, self.hidden_dim // 2), -1, 1)
                      for _ in range(2)]

        return tuple(hidden)

    def __get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        length = sentence.shape[1]
        embeddings = self.word_embeddings(sentence).view(self.batch_size, length, self.embedding_dim)

        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        lstm_out = lstm_out.view(self.batch_size, -1, self.hidden_dim)
        logits = self.hidden2tag(lstm_out)
        #logits = torch.nn.Softmax()(logits)

        return logits

    def real_path_score(self, logits, label):
        '''
        caculate real path score  
        :params logits -> [len_sent * tag_size]
        :params label  -> [1 * len_sent]

        Score = Emission_Score + Transition_Score  
        Emission_Score = logits(0, label[START]) + logits(1, label[1]) + ... + logits(n, label[STOP])  
        Transition_Score = Trans(label[START], label[1]) + Trans(label[1], label[2]) + ... + Trans(label[n-1], label[STOP])  
        '''
        if self.use_cuda:
            score = torch.zeros(1).cuda()
            label = torch.cat([torch.cuda.LongTensor([self.tag_map[START_TAG]]), label])
        else:
            score = torch.zeros(1)
            label = torch.cat([torch.LongTensor([self.tag_map[START_TAG]]), label])
        for index, logit in enumerate(logits):
            emission_score = logit[label[index + 1]]
            transition_score = self.transitions[label[index], label[index + 1]]
            score += emission_score + transition_score

        score += self.transitions[label[-1], self.tag_map[STOP_TAG]]

        return score

    def total_score(self, logits, label):
        """
        caculate total score

        :params logits -> [len_sent * tag_size]
        :params label  -> [1 * tag_size]

        SCORE = log(e^S1 + e^S2 + ... + e^SN)
        """
        obs = []
        if self.use_cuda:
            previous = torch.full((1, self.tag_size), 0).cuda()
        else:
            previous = torch.full((1, self.tag_size), 0)
        for index in range(len(logits)): 
            previous = previous.expand(self.tag_size, self.tag_size).t()
            obs = logits[index].view(1, -1).expand(self.tag_size, self.tag_size)
            scores = previous + obs + self.transitions
            previous = log_sum_exp(scores)
        previous = previous + self.transitions[:, self.tag_map[STOP_TAG]]
        # caculate total_scores
        total_scores = log_sum_exp(previous.t())[0]
        return total_scores

    def neg_log_likelihood(self, sentences, tags, length):
        self.batch_size = sentences.size(0)
        logits = self.__get_lstm_features(sentences)
        if self.use_cuda:
            real_path_score = torch.zeros(1).cuda()
            total_score = torch.zeros(1).cuda()
        else:
            real_path_score = torch.zeros(1)
            total_score = torch.zeros(1)

        for logit, tag, leng in zip(logits, tags, length):
            logit = logit[:leng]
            #print(logit)
            tag = tag[:leng]

            real_path_score += self.real_path_score(logit, tag)
            total_score += self.total_score(logit, tag)

        return total_score - real_path_score


    def forward(self, sentences, lengths=None):
        """
        :params sentences sentences to predict
        :params lengths represent the ture length of sentence, the default is sentences.size(-1)
        """
        if self.use_cuda:
            sentences = torch.tensor(sentences, dtype=torch.long).cuda()
        else:
            sentences = torch.tensor(sentences, dtype=torch.long)

        if not lengths:
            lengths = [i.size(-1) for i in sentences]
        self.batch_size = sentences.size(0)
        logits = self.__get_lstm_features(sentences)
        self.lstm_logit = logits
        scores = []
        paths = []
        for logit, leng in zip(logits, lengths):
            logit = logit[:leng]
            score, path, probs = self.__viterbi_decode(logit)
            scores.append(score)
            paths.append(path)
        return scores, paths, probs

    def __viterbi_decode(self, logits):

        if self.use_cuda:
            trellis = torch.zeros(logits.size()).cuda()
            backpointers = torch.zeros(logits.size(), dtype=torch.long).cuda()
        else:
            trellis = torch.zeros(logits.size())
            backpointers = torch.zeros(logits.size(), dtype=torch.long)

        trellis[0] = logits[0]
        for t in range(1, len(logits)):
            v = trellis[t - 1].unsqueeze(1).expand_as(self.transitions) + self.transitions
            trellis[t] = logits[t] + torch.max(v, 0)[0]
            backpointers[t] = torch.max(v, 0)[1]

        viterbi = [torch.max(trellis[-1], -1)[1].cpu().tolist()]
        backpointers = backpointers.cpu().numpy()

        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()

        probs = [torch.nn.Softmax()(trellis[i])[viterbi[i]].detach().tolist() for i in range(len(viterbi))]

        viterbi_score = torch.max(trellis[-1], 0)[0].cpu().tolist()
        return viterbi_score, viterbi, probs
