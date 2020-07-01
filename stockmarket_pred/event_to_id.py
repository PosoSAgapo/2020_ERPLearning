import sys
sys.path.insert(0, '/users4/bwchen/CommonsenseERL_EMNLP_2019/')

import torch
import numpy as np
from event_tensors.glove_utils import Glove
import pickle

def read_reverb_zpar_data(filename):
    date_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            event, date = line.strip().split(' || ')
            if date not in date_dict:
                date_dict[date] = []
            date_dict[date].append(event)
    return date_dict


def read_reverb_data(filename):
    date_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split(' || ')
            date = line[2]
            events = line[3:]
            events = [event for event in events if len(event.split('|')) == 3]
            if len(events) > 0:
                if date not in date_dict:
                    date_dict[date] = []
                date_dict[date] += events

    return date_dict


def event_to_id(date_dict, glove, max_phrase_size=10):
    subj_oov_count = 0
    verb_oov_count = 0
    obj_oov_count = 0
    result = {}
    event_d = {}
    for date in date_dict:
        event_d[date]=''
        events = date_dict[date]
        all_subj_id = []
        all_subj_w = []
        all_verb_id = []
        all_verb_w = []
        all_obj_id = []
        all_obj_w = []
        for event in events:
            subj, verb, obj = event.lower().split('|')
            subj_words = subj.split(' ')
            verb_words = verb.split(' ')
            obj_words = obj.split(' ')

            subj_id, subj_w = glove.transform(subj_words, max_phrase_size)
            verb_id, verb_w = glove.transform(verb_words, max_phrase_size)
            obj_id, obj_w = glove.transform(obj_words, max_phrase_size)
            if subj_id is None:
                subj_id = np.zeros(max_phrase_size)
                subj_w = np.zeros(max_phrase_size)
                subj_oov_count += 1
            if verb_id is None:
                verb_id = np.zeros(max_phrase_size)
                verb_w = np.zeros(max_phrase_size)
                verb_oov_count += 1
            if obj_id is None:
                obj_id = np.zeros(max_phrase_size)
                obj_w = np.zeros(max_phrase_size)
                obj_oov_count += 1

            # subj_words = [word for word in subj_words if word in glove.__dict__]
            # verb_words = [word for word in verb_words if word in glove.__dict__]
            # obj_words = [word for word in obj_words if word in glove.__dict__]
            # if len(subj_words) == 0:
            #     subj_id = np.zeros(max_phrase_size)
            #     subj_w = np.zeros(max_phrase_size)
            #     subj_oov_count += 1
            # else:
            #     subj_id, subj_w = glove.transform(subj_words, max_phrase_size)
            # if len(verb_words) == 0:
            #     verb_id = np.zeros(max_phrase_size)
            #     verb_w = np.zeros(max_phrase_size)
            #     verb_oov_count += 1
            # else:
            #     verb_id, verb_w = glove.transform(verb_words, max_phrase_size)
            # if len(obj_words) == 0:
            #     obj_id = np.zeros(max_phrase_size)
            #     obj_w = np.zeros(max_phrase_size)
            #     obj_oov_count += 1
            # else:
            #     obj_id, obj_w = glove.transform(obj_words, max_phrase_size)

            all_subj_id.append(subj_id)
            all_subj_w.append(subj_w)
            all_verb_id.append(verb_id)
            all_verb_w.append(verb_w)
            all_obj_id.append(obj_id)
            all_obj_w.append(obj_w)
        all_subj_id = torch.LongTensor(all_subj_id)
        all_subj_w = torch.FloatTensor(all_subj_w)
        all_verb_id = torch.LongTensor(all_verb_id)
        all_verb_w = torch.FloatTensor(all_verb_w)
        all_obj_id = torch.LongTensor(all_obj_id)
        all_obj_w = torch.FloatTensor(all_obj_w)
        result[date] = (all_subj_id, all_subj_w, all_verb_id, all_verb_w, all_obj_id, all_obj_w)
        event_d[date] = event_d[date]+'///'+ subj+' '+verb+' '+obj
    return result, subj_oov_count, verb_oov_count, obj_oov_count,event_d


def main():
    # output_file = 'data/cooked/reverb_zpar.pt'
    output_file = 'data/cooked/reverb.pt1'

    date_dict = {}
    # train_data = read_reverb_zpar_data('data/cooked/reverb_zpar_train.txt')
    # dev_data = read_reverb_zpar_data('data/cooked/reverb_zpar_dev.txt')
    # test_data = read_reverb_zpar_data('data/cooked/reverb_zpar_test.txt')
    train_data = read_reverb_data('data/cooked/reverb_train.txt')
    dev_data= read_reverb_data('data/cooked/reverb_dev.txt')
    test_data = read_reverb_data('data/cooked/reverb_test.txt')
    date_dict.update(train_data)
    date_dict.update(dev_data)
    date_dict.update(test_data)

    glove = Glove('/users4/bwchen/CommonsenseERL_EMNLP_2019/data/glove.6B.100d.ext.txt')
    id_data, subj_oov_count, verb_oov_count, obj_oov_count,event_d = event_to_id(date_dict, glove)

    print('subj oov:', subj_oov_count)
    print('verb oov:', verb_oov_count)
    print('obj oov :', obj_oov_count)

    id2word = glove.reverse_dict()
    date = '2006-10-20'
    subj_id, subj_w, verb_id, verb_w, obj_id, obj_w = id_data[date]
    print('subj:', ' '.join([id2word[int(i)] for i in subj_id[0]]))
    print('verb:', ' '.join([id2word[int(i)] for i in verb_id[0]]))
    print('obj: ', ' '.join([id2word[int(i)] for i in obj_id[0]]))
    #exit()
    print(event_d)
    torch.save(id_data, output_file)
    f=open('event_data','wb')
    pickle.dump(event_d,f)
    f.close()


if __name__ == '__main__':
    main()
