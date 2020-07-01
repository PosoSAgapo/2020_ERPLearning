import sys
# 修改为事件表示模型所在的目录，需要import那个目录下的代码
sys.path.insert(0, '/users4/bwchen/CommonsenseERL_EMNLP_2019/')

import torch
import torch.nn as nn
import argparse
from model_yago import NeuralTensorNetwork, LowRankNeuralTensorNetwork


def id_to_emb(id_data, model, batch_size=128):
    result = {}
    for date in id_data:
        subj_id, subj_w, verb_id, verb_w, obj_id, obj_w = id_data[date]
        subj_id = subj_id.cuda()
        subj_w = subj_w.cuda()
        verb_id = verb_id.cuda()
        verb_w = verb_w.cuda()
        obj_id = obj_id.cuda()
        obj_w = obj_w.cuda()
        emb = model(subj_id, subj_w, verb_id, verb_w, obj_id, obj_w)
        result[date] = torch.mean(emb, dim=0).detach().cpu()
    return result


def main(model,model_file,em_r):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='/users4/bwchen/CommonsenseERL_EMNLP_2019/stockmarket_pred/data/cooked/reverb.pt')
    parser.add_argument('--output_file', type=str, default='/users4/bwchen/CommonsenseERL_EMNLP_2019/stockmarket_pred/data/emb/reverb.pt')
    parser.add_argument('--model', type=str, default=model)
    parser.add_argument('--model_file', type=str, default=model_file)
    parser.add_argument('--vocab_size', type=int, default=400000)
    parser.add_argument('--emb_dim', type=int, default=100)
    parser.add_argument('--em_k', type=int, default=100)
    parser.add_argument('--em_r', type=int, default=em_r)
    option = parser.parse_args()

    input_file = option.input_file
    output_file = option.output_file
    model_file = option.model_file

    id_data = torch.load(input_file)

    embeddings = nn.Embedding(option.vocab_size, option.emb_dim)
    if option.model == 'NTN':
        model = NeuralTensorNetwork(embeddings, option.em_k)
    elif option.model == 'LowRankNTN':
        model = LowRankNeuralTensorNetwork(embeddings, option.em_k, option.em_r)
    else:
        print('Unknown model type: ' + option.model)
        exit(1)
    state_dict = torch.load(model_file)
    if 'event_model_state_dict' in state_dict:
        state_dict = state_dict['event_model_state_dict']
    elif 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model.cuda()

    emb_data = id_to_emb(id_data, model)
    torch.save(emb_data, output_file)    


if __name__ == '__main__':
    main(model,model_file,em_r)
