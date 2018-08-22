from ConvReducer import ConvReducer
from FNCLoader import FNCDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from InferSent_master.models import InferSent

class StanceDetector(nn.Module):
    def __init__(self, reducer):
        super(StanceDetector, self).__init__()
        self.reducer = reducer
        self.fc_headline = nn.Linear(128, 32)
        self.fc_body = nn.Linear(128, 32)
        self.relu = nn.ReLU()
        self.fc_output = nn.Linear(64, 4)
        self.softmax = nn.Softmax()
        
    def forward(self, body, headline):
        body_features = self.reducer(body).sum(dim=-2) / len(body)
        headline_features = self.reducer(headline.unsqueeze(0))[0].cuda()
        reduced_b_feats = self.fc_body(self.relu(body_features))
        reduced_hl_feats = self.fc_headline(self.relu(headline_features)) #.cuda()
        concatenated = torch.cat([reduced_b_feats, reduced_hl_feats], dim = -1)
        out = F.log_softmax(self.fc_output(concatenated))
        return out

if __name__ == '__main__':
    MODEL_PATH =  '/home/rmacias3/Desktop/stance_detection/InferSent_master/encoder/infersent2.pkl'
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))
    W2V_PATH = '/home/rmacias3/Desktop/stance_detection/InferSent_master/dataset/fastText/crawl-300d-2M.vec'
    model.set_w2v_path(W2V_PATH)
    print('building vocabulary...')
    model.build_vocab_k_words(K=100000)
    print('building dataset...')
    fnc_path = '/home/rmacias3/Desktop/stance_detection/fnc-1-master/'
    article_csv, stance_csv = fnc_path + 'train_bodies.csv', fnc_path + 'train_stances.csv'
    data_loader = FNCDataset(article_csv, stance_csv, model)
    
    reducer = ConvReducer().cuda()
    stance_detector = StanceDetector(reducer).cuda()

    print('training...')
    losses = []
    #.1 weight for unrelated, 1 weight for agree, 1 weight for discuss, 1 weight for disagree
    weights = torch.Tensor([0, 1.0, 1.0, 1.0])
    optimizer = optim.Adam(stance_detector.parameters())
    loss_function = nn.NLLLoss(weight=weights.cuda())
    for i in range(len(data_loader)):
        cur_pair = data_loader[i]
        print('at body: ', i)
        body, headline_and_stance = cur_pair['body_embeds'], list(cur_pair['hl_embeds_and_stance'])
        for hl_embed, gt_stance in headline_and_stance:
            #print(body, hl_embed)
            stance_detector.zero_grad()
            stance_scores = stance_detector(body.cuda(), hl_embed.cuda())
    #         print(gt_stance.unsqueeze(0).type(torch.LongTensor), 'gt_stance')
    #         print(stance_scores.unsqueeze(0), 'output')
            loss = loss_function(stance_scores.unsqueeze(0).cuda(), gt_stance.type(torch.LongTensor).cuda())
            loss.backward()
            optimizer.step()
            if gt_stance[0] != 0:
                print(loss, gt_stance, stance_scores)
            losses.append(loss)