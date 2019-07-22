import torch
import torch.optim as optim


from model import BiLSTMCRF
from utils import f1_score, DataManager, JsonConfig
import json
import numpy as np
from sklearn.metrics import accuracy_score


def evaluate():
    sentences, labels, length = zip(*dev_batch.__next__())
    _, paths, _  = model(sentences)

    print("\teval")
    for tag in args.tags:
        f1_score(labels, paths, tag, model.tag_map, length)
    
    acc = 0
    for i, (l, p) in enumerate(zip(labels, paths)):
        acc += accuracy_score(l[:length[i]], p[:length[i]])
    print(round(acc / (len(paths) + 1e-5),3))


def train():
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.98)

    for epoch in range(args.num_epoch):
        index = 0
        for batch in train_manager.get_batch():
            index += 1
            model.zero_grad()

            sentences, tags, length = zip(*batch)
            
            if args.cuda:
                sentences_tensor = torch.tensor(sentences, dtype=torch.long).cuda()
                tags_tensor = torch.tensor(tags, dtype=torch.long).cuda()
                length_tensor = torch.tensor(length, dtype=torch.long).cuda()
            else:
                sentences_tensor = torch.tensor(sentences, dtype=torch.long)
                tags_tensor = torch.tensor(tags, dtype=torch.long)
                length_tensor = torch.tensor(length, dtype=torch.long)

            loss = model.neg_log_likelihood(sentences_tensor, tags_tensor, length_tensor)

            progress = ("█" * int(index * 25 / total_size)).ljust(25)
            print("""epoch [{}] |{}| {}/{}\n\tloss {:.2f}""".format(
                epoch+1, progress, index, total_size, loss.cpu().tolist()[0]/args.batch_size
            )
            )

            print("-" * 50)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 50 == 0:
            torch.save(model.state_dict(), './' + 'BIO_BiLSTMCRF_Focal_v1.0_'+str(epoch+1)+'.pkl')
        evaluate()


if __name__ == '__main__':

    params = JsonConfig('./params_BiLSTM.json')
    args = params.values
    args.cuda = eval(args.cuda)

    # load tag_map
    with open(args.tag_map_path, 'r', encoding='utf-8') as file:
        tag_map = json.load(file)

    args.tag_map = tag_map
    tags = [tag[2:] for tag in tag_map.keys() if '<' in tag]
    tags = list(np.unique(tags))
    args.tags = tags

    train_manager = DataManager(args)
    total_size = len(train_manager.batch_data)

    dev_manager = DataManager(args, data_type="dev")
    dev_batch = dev_manager.iteration()

    model =  BiLSTMCRF(
        tag_map=train_manager.tag_map,
        batch_size=args.batch_size,
        vocab_size=train_manager.tokenizer.get_piece_size(),
        dropout=args.dropout,
        embedding_dim=args.embedding_size,
        hidden_dim=args.hidden_size,
        num_layer=args.num_layer,
        use_cuda=args.cuda
    )

    if args.cuda:
        model = model.cuda()

    train()