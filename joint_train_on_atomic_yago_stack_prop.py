import torch
import torch.nn as nn
import torch.utils.data
import sys
import logging
import argparse
import itertools
from model import NeuralTensorNetwork, LowRankNeuralTensorNetwork, RoleFactoredTensorModel, BiLSTMEncoder, MarginLoss, NeuralTensorNetwork_yago
from dataset import EventIntentSentimentDataset, EventIntentSentimentDataset_collate_fn, YagoDataset, YagoDataset_collate_fn
from event_tensors.glove_utils import Glove


def run_batch(option, batch, event_model, intent_model, event_scorer, intent_scorer, sentiment_classifier, criterion, sentiment_criterion):
    subj_id, subj_w, verb_id, verb_w, obj_id, obj_w, neg_obj_id, neg_obj_w, \
    intent, neg_intent, sentiment, neg_sentiment = batch
    if option.use_gpu:
        subj_id = subj_id.cuda()
        subj_w = subj_w.cuda()
        verb_id = verb_id.cuda()
        verb_w = verb_w.cuda()
        obj_id = obj_id.cuda()
        obj_w = obj_w.cuda()
        neg_obj_id = neg_obj_id.cuda()
        neg_obj_w = neg_obj_w.cuda()
        intent = intent.cuda()
        neg_intent = neg_intent.cuda()
        sentiment = sentiment.cuda()
        neg_sentiment = neg_sentiment.cuda()
    # event loss
    pos_event_emb = event_model(subj_id, subj_w, verb_id, verb_w, obj_id, obj_w)
    neg_event_emb = event_model(subj_id, subj_w, verb_id, verb_w, neg_obj_id, neg_obj_w)
    pos_event_score = event_scorer(pos_event_emb).squeeze()
    neg_event_score = event_scorer(neg_event_emb).squeeze()
    loss_e = criterion(pos_event_score, neg_event_score)
    # sentiment loss
    sentiment_pred = sentiment_classifier(pos_event_emb)
    loss_s = sentiment_criterion(sentiment_pred.squeeze(1), sentiment.float())
    # intent loss
    event_emb = torch.cat((pos_event_emb, sentiment_pred), dim=1)
    pos_intent_emb = intent_model(intent)
    neg_intent_emb = intent_model(neg_intent)
    pos_intent_score = intent_scorer(event_emb, pos_intent_emb)
    neg_intent_score = intent_scorer(event_emb, neg_intent_emb)
    loss_i = criterion(pos_intent_score, neg_intent_score)
    # total loss
    alpha1 = option.alpha1
    alpha2 = option.alpha2
    alpha3 = (1 - alpha1 - alpha2) if option.alpha3 is None else option.alpha3
    loss = alpha1 * loss_e + alpha2 * loss_i + alpha3 * loss_s
    return loss, loss_e, loss_i, loss_s


def run_yago_batch(option, batch, event_model, relation_models, criterion, relation_scorer, event_scorer):
    subj_id, subj_w, verb_id, verb_w, obj_id, obj_w, neg_obj_id, neg_obj_w,subj_relation, subj_attr_id, subj_attr_w, obj_relation, obj_attr_id, obj_attr_w, neg_attr_id, neg_attr_w=batch
    if option.use_gpu:
        subj_id = subj_id.cuda()
        subj_w = subj_w.cuda()
        verb_id = verb_id.cuda()
        verb_w = verb_w.cuda()
        obj_id = obj_id.cuda()
        obj_w = obj_w.cuda()
        neg_obj_id = neg_obj_id.cuda()
        neg_obj_w = neg_obj_w.cuda()
        subj_attr_id = subj_attr_id.cuda()
        subj_attr_w = subj_attr_w.cuda()
        obj_attr_id = obj_attr_id.cuda()
        obj_attr_w = obj_attr_w.cuda()
        neg_attr_id = neg_attr_id.cuda()
        neg_attr_w = neg_attr_w.cuda()
    pos_event_emb = event_model(subj_id, subj_w, verb_id, verb_w, obj_id, obj_w)#事件
    neg_event_emb = event_model(subj_id, subj_w, verb_id, verb_w, neg_obj_id, neg_obj_w)#负事件
    subj_relation_model = relation_models[subj_relation]#找到关系模型
    obj_relation_model = relation_models[obj_relation]#找到关系模型
    subj_attr_relation_emb = subj_relation_model(subj_id,subj_w,subj_attr_id,subj_attr_w)#主语与关系
    obj_attr_relation_emb = obj_relation_model(obj_id,obj_w,obj_attr_id,obj_attr_w)#obj与关系
    neg_obj_attr_relation_emb = obj_relation_model(obj_id,obj_w,neg_attr_id,neg_attr_w)#负主语与关系
    pos_event_score  = event_scorer(pos_event_emb).squeeze()
    neg_event_score  = event_scorer(neg_event_emb).squeeze()
    loss_e = criterion(pos_event_score,neg_event_score)#正负事件打分
    subj_attr_score=relation_scorer(subj_attr_relation_emb).squeeze()#主语关系得分
    obj_attr_score = relation_scorer(obj_attr_relation_emb).squeeze()#宾语关系得分
    neg_obj_attr_score  = relation_scorer(neg_obj_attr_relation_emb).squeeze()#负宾语关系得分
    loss_attr = criterion(obj_attr_score,neg_obj_attr_score)#两个负宾语打分
    alpha4 = option.alpha4
    alpha5 = (1 - alpha4) if option.alpha5 is None else option.alpha5
    loss_sum = alpha4 * loss_e + alpha5 * loss_attr
    return loss_sum,loss_e,loss_attr


def main(option):
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    torch.manual_seed(option.random_seed)

    glove = Glove(option.emb_file)
    logging.info('Embeddings loaded')

    train_dataset = EventIntentSentimentDataset()
    logging.info('Loading train dataset: ' + option.train_dataset)
    train_dataset.load(option.train_dataset, glove)
    logging.info('Loaded train dataset: ' + option.train_dataset)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=EventIntentSentimentDataset_collate_fn, batch_size=option.batch_size, shuffle=True)

    if option.dev_dataset is not None:
        dev_dataset = EventIntentSentimentDataset()
        logging.info('Loading dev dataset: ' + option.dev_dataset)
        dev_dataset.load(option.dev_dataset, glove)
        logging.info('Loaded dev dataset: ' + option.dev_dataset)
        dev_data_loader = torch.utils.data.DataLoader(dev_dataset, collate_fn=EventIntentSentimentDataset_collate_fn, batch_size=len(dev_dataset), shuffle=False)

    yago_train_dataset = YagoDataset()
    logging.info('Loading YAGO train dataset: ' + option.yago_train_dataset)
    yago_train_dataset.load(option.yago_train_dataset, glove)
    yago_train_data_loader = torch.utils.data.DataLoader(yago_train_dataset, collate_fn=YagoDataset_collate_fn, batch_size=option.batch_size, shuffle=True)

    if option.yago_dev_dataset is not None:
        yago_dev_dataset = YagoDataset()
        logging.info('Loading YAGO dev dataset: ' + option.yago_dev_dataset)
        yago_dev_dataset.load(option.yago_dev_dataset, glove)
        yago_dev_data_loader = torch.utils.data.DataLoader(yago_dev_dataset, collate_fn=YagoDataset_collate_fn, batch_size=len(yago_dev_dataset), shuffle=False)

    embeddings = nn.Embedding(option.vocab_size, option.emb_dim, padding_idx=1)
    if option.model_type == 'NTN':
        event_model = NeuralTensorNetwork(embeddings, option.em_k)
    elif option.model_type == 'RoleFactor':
        event_model = RoleFactoredTensorModel(embeddings, option.em_k)
    elif option.model_type == 'LowRankNTN':
        event_model = LowRankNeuralTensorNetwork(embeddings, option.em_k, option.em_r)

    intent_model = BiLSTMEncoder(embeddings, option.im_hidden_size, option.im_num_layers)

    relation_models = {}
    for relation in yago_train_dataset.relations:
        relation_model = NeuralTensorNetwork_yago(embeddings, option.em_k)
        relation_models[relation] = relation_model

    if option.scorer_actv_func == 'sigmoid':
        scorer_actv_func = nn.Sigmoid
    elif option.scorer_actv_func == 'relu':
        scorer_actv_func = nn.ReLU
    elif option.scorer_actv_func == 'tanh':
        scorer_actv_func = nn.Tanh
    event_scorer = nn.Sequential(nn.Linear(option.em_k, 1), scorer_actv_func())
    relation_scorer = nn.Sequential(nn.Linear(option.em_k, 1), scorer_actv_func())

    intent_scorer = nn.CosineSimilarity(dim=1)
    sentiment_classifier = nn.Linear(option.em_k, 1)
    criterion = MarginLoss(option.margin)
    sentiment_criterion = nn.BCEWithLogitsLoss()
    yago_criterion = MarginLoss(option.yago_margin)

    # load pretrained embeddings
    embeddings.weight.data.copy_(torch.from_numpy(glove.embd).float())

    if not option.update_embeddings:
        event_model.embeddings.weight.requires_grad = False

    if option.use_gpu:
        event_model.cuda()
        intent_model.cuda()
        sentiment_classifier.cuda()
        event_scorer.cuda()
        relation_scorer.cuda()
        for relation_model in relation_models.values():
            relation_model.cuda()

    embeddings_param_id = [id(param) for param in embeddings.parameters()]
    params = [
        { 'params': embeddings.parameters() },
        { 'params': [param for param in event_model.parameters() if id(param) not in embeddings_param_id], 'weight_decay': option.weight_decay },
        { 'params': [param for param in event_scorer.parameters() if id(param) not in embeddings_param_id], 'weight_decay': option.weight_decay },
        { 'params': [param for param in intent_model.parameters() if id(param) not in embeddings_param_id], 'weight_decay': option.weight_decay },
        { 'params': [param for param in sentiment_classifier.parameters() if id(param) not in embeddings_param_id], 'weight_decay': option.weight_decay }
    ]
    for relation in relation_models:
        params.append({ 'params': [param for param in relation_models[relation].parameters() if id(param) not in embeddings_param_id], 'weight_decay': option.weight_decay })
    optimizer = torch.optim.Adagrad(params, lr=option.lr)

    # load checkpoint if provided:
    if option.load_checkpoint != '':
        checkpoint = torch.load(option.load_checkpoint)
        event_model.load_state_dict(checkpoint['event_model_state_dict'])
        intent_model.load_state_dict(checkpoint['intent_model_state_dict'])
        event_scorer.load_state_dict(checkpoint['event_scorer_state_dict'])
        sentiment_classifier.load_state_dict(checkpoint['sentiment_classifier_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for relation in relation_models:
            relation_models[relation].load_state_dict(checkpoint['relation_model_state_dict'][relation])
        logging.info('Loaded checkpoint: ' + option.load_checkpoint)
    # load pretrained event model instead:
    elif option.pretrained_event_model != '':
        checkpoint = torch.load(option.pretrained_event_model)
        event_model.load_state_dict(checkpoint['model_state_dict'])
        logging.info('Loaded pretrained event model: ' + option.pretrained_event_model)

    for epoch in range(option.epochs):
        epoch += 1
        logging.info('Epoch ' + str(epoch))

        # train set
        # train set
        avg_loss_e = 0
        avg_loss_i = 0
        avg_loss_s = 0
        avg_loss = 0
        avg_loss_sum = 0
        avg_loss_event = 0
        avg_loss_attr = 0
        k=0
        # assume yago dataset is larger than atomic
        atomic_iterator = itertools.cycle(iter(train_data_loader))
        yago_iterator = iter(yago_train_data_loader)
        # iterate over yago dataset (atomic dataset is cycled)
        for i, yago_batch in enumerate(yago_iterator):
            atomic_batch = next(atomic_iterator)

            optimizer.zero_grad()
            loss, loss_e, loss_i, loss_s = run_batch(option, atomic_batch, event_model, intent_model, event_scorer, intent_scorer,sentiment_classifier, criterion, sentiment_criterion)
            loss.backward()
            optimizer.step()
            avg_loss_e += loss_e.item() / option.report_every
            avg_loss_i += loss_i.item() / option.report_every
            avg_loss_s += loss_s.item() / option.report_every
            avg_loss += loss.item() / option.report_every
            if i % option.report_every==0:
                logging.info('Atomic batch %d, loss_e=%.4f, loss_i=%.4f, loss_s=%.4f, loss=%.4f' % (i, avg_loss_e, avg_loss_i, avg_loss_s, avg_loss))
                avg_loss_e = 0
                avg_loss_i = 0
                avg_loss_s = 0
                avg_loss = 0

            optimizer.zero_grad()
            loss_sum,loss_event,loss_attr = run_yago_batch(option, yago_batch, event_model,relation_models,yago_criterion,relation_scorer,event_scorer)
            loss_sum.backward()
            optimizer.step()
            avg_loss_sum += loss_sum.item() / option.report_every
            avg_loss_event += loss_event.item() / option.report_every
            avg_loss_attr += loss_attr.item() / option.report_every
            if i % option.report_every==0:
                logging.info('YAGO batch %d, loss_event=%.4f, loss_attr=%.4f, loss=%.4f' % (i, avg_loss_event, avg_loss_attr, avg_loss_sum))
                avg_loss_sum = 0
                avg_loss_event = 0
                avg_loss_attr = 0

        # dev set
        if option.dev_dataset is not None:
            event_model.eval()
            intent_model.eval()
            event_scorer.eval()
            sentiment_classifier.eval()
            batch = next(iter(dev_data_loader))
            with torch.no_grad():
                loss, loss_e, loss_i, loss_s = run_batch(option, batch, event_model, intent_model, event_scorer, intent_scorer, sentiment_classifier, criterion, sentiment_criterion)
            logging.info('Eval on dev set, loss_e=%.4f, loss_i=%.4f, loss_s=%.4f, loss=%.4f' % (loss_e.item(), loss_i.item(), loss_s.item(), loss.item()))
            event_model.train()
            intent_model.train()
            event_scorer.train()
            sentiment_classifier.train()

        # dev set (yago)
        if option.yago_dev_dataset is not None:
            for key in relation_models.keys():
                relation_models[key].eval()
            relation_scorer.eval()
            event_model.eval()
            yago_dev_batch = next(iter(yago_dev_data_loader))
            with torch.no_grad():
                loss_sum, loss_event, loss_attr = run_yago_batch(option, yago_dev_batch, event_model, relation_models, criterion_yago,relation_scorer,event_scorer)
            logging.info('Eval on yago dev set, loss_sum=%.4f, loss_event=%.4f, loss_attr=%.4f, ' % (loss_sum.item(), loss_event.item(), loss_attr.item()))
            for key in relation_models.keys():
                relation_models[key].train()
            relation_scorer.train()

        if option.save_checkpoint != '':
            checkpoint = {
                'event_model_state_dict': event_model.state_dict(),
                'intent_model_state_dict': intent_model.state_dict(),
                'event_scorer_state_dict': event_scorer.state_dict(),
                'sentiment_classifier_state_dict': sentiment_classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'relation_model_state_dict': { relation : relation_models[relation].state_dict() for relation in relation_models }
            }
            torch.save(checkpoint, option.save_checkpoint + '_' + str(epoch))
            logging.info('Saved checkpoint: ' + option.save_checkpoint + '_' + str(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=19960125)
    parser.add_argument('--vocab_size', type=int, default=400000)
    parser.add_argument('--emb_dim', type=int, default=100)
    parser.add_argument('--update_embeddings', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--emb_file', type=str, default='data/glove.6B.100d.ext.txt')
    parser.add_argument('--train_dataset', type=str, default='data/atomic/all2.txt')
    parser.add_argument('--dev_dataset', type=str, default=None)
    parser.add_argument('--yago_train_dataset', type=str, default='data/yago_nyt.txt')
    parser.add_argument('--yago_dev_dataset', type=str, default=None)
    parser.add_argument('--model_type', type=str, choices=['NTN', 'LowRankNTN'], default='NTN')
    parser.add_argument('--pretrained_event_model', type=str, default='model/pretrain_nyt/ntn/NeuralTensorNetwork_2007.pt')
    parser.add_argument('--scorer_actv_func', type=str, choices=['sigmoid', 'tanh', 'relu'], default='sigmoid')
    parser.add_argument('--em_k', type=int, default=100)
    parser.add_argument('--em_r', type=int, default=10)
    parser.add_argument('--im_hidden_size', type=int, default=101)
    parser.add_argument('--im_num_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--margin', type=float, default=0.5)
    parser.add_argument('--yago_margin', type=int, default=1)
    parser.add_argument('--alpha1', type=float, default=0.33)
    parser.add_argument('--alpha2', type=float, default=0.33)
    parser.add_argument('--alpha3', type=float, default=None)
    parser.add_argument('--alpha4', type=float, default=0.5)
    parser.add_argument('--alpha5', type=float, default=0.5)
    parser.add_argument('--report_every', type=int, default=50)
    parser.add_argument('--load_checkpoint', type=str, default='')
    parser.add_argument('--save_checkpoint', type=str, default='')
    option = parser.parse_args()

    main(option)
