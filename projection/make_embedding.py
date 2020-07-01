import torch
from torch.nn import Embedding
import sys
from argparse import ArgumentParser

sys.path.insert(0, '/users5/kliao/code/CommonsenseERL_EMNLP_2019')
from event_tensors.glove_utils import Glove
from model import NeuralTensorNetwork, LowRankNeuralTensorNetwork


def main(option):
    glove = Glove(option.emb_file)
    print(option.emb_file + ' loaded')

    embedding = Embedding(option.vocab_size, option.emb_dim, padding_idx=1)
    if option.model_type == 'NTN':
        model = NeuralTensorNetwork(embedding, option.em_k)
    elif option.model_type == 'LowRankNTN':
        model = LowRankNeuralTensorNetwork(embedding, option.em_k, option.em_r)

    checkpoint = torch.load(option.model_file, map_location='cpu')
    if type(checkpoint) == dict:
        if 'event_model_state_dict' in checkpoint:
            state_dict = checkpoint['event_model_state_dict']
        else:
            state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    print(option.model_file + ' loaded')
    model.eval()
    model.to(option.device)

    all_subj_id = []
    all_subj_w = []
    all_verb_id = []
    all_verb_w = []
    all_obj_id = []
    all_obj_w = []
    all_labels = []
    all_event_texts = []
    for label, filename in enumerate(option.input_files):
        print('loading ' + filename)
        lines = open(filename, 'r').readlines()
        for line in lines:
            subj, verb, obj = line.lower().strip().split(' | ')
            event_text = '(' + subj + ', ' + verb + ', ' + obj + ')'
            subj = subj.split(' ')
            verb = verb.split(' ')
            obj = obj.split(' ')
            subj_id, subj_w = glove.transform(subj, 10)
            verb_id, verb_w = glove.transform(verb, 10)
            obj_id, obj_w = glove.transform(obj, 10)
            if subj_id is not None and verb_id is not None and obj_id is not None and event_text not in all_event_texts:
                all_subj_id.append(subj_id)
                all_subj_w.append(subj_w)
                all_verb_id.append(verb_id)
                all_verb_w.append(verb_w)
                all_obj_id.append(obj_id)
                all_obj_w.append(obj_w)
                all_labels.append(label)
                all_event_texts.append(event_text)
    
    all_subj_id = torch.tensor(all_subj_id, dtype=torch.long, device=option.device)
    all_subj_w = torch.tensor(all_subj_w, dtype=torch.float, device=option.device)
    all_verb_id = torch.tensor(all_verb_id, dtype=torch.long, device=option.device)
    all_verb_w = torch.tensor(all_verb_w, dtype=torch.float, device=option.device)
    all_obj_id = torch.tensor(all_obj_id, dtype=torch.long, device=option.device)
    all_obj_w = torch.tensor(all_obj_w, dtype=torch.float, device=option.device)
    all_event_embeddings = model(all_subj_id, all_subj_w, all_verb_id, all_verb_w, all_obj_id, all_obj_w).detach().cpu()

    torch.save({
        'embeddings': all_event_embeddings,
        'labels': torch.tensor(all_labels, dtype=torch.long),
        'event_texts': all_event_texts
    }, option.output_file)
    print('saved to ' + option.output_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--emb_file', type=str, default='/users5/kliao/code/CommonsenseERL_EMNLP_2019/data/glove.6B.100d.ext.txt')
    parser.add_argument('--model_type', type=str, choices=['NTN', 'LowRankNTN'], default='LowRankNTN')
    parser.add_argument('--vocab_size', type=int, default=400000)
    parser.add_argument('--emb_dim', type=int, default=100)
    parser.add_argument('--em_k', type=int, default=100)
    parser.add_argument('--em_r', type=int, default=10)
    parser.add_argument('--model_file', type=str, default='/users5/kliao/code/CommonsenseERL_EMNLP_2019/model/publish_2020/yago_stack_prop_lowrank10_batch128_sigmoid_0.3_0.3_e9_b708_hard_79.13.pt')
    parser.add_argument('--input_files', type=str, nargs='+', default=['data/food.txt', 'data/learn.txt', 'data/violence.txt', 'data/sports.txt'])
    parser.add_argument('--output_file', type=str, default='embedding.pt')
    option = parser.parse_args()

    main(option)
