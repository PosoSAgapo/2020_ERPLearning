import os
import sys
import random
import logging
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader

from seq2seq.models import EncoderRNN, DecoderRNN
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.trainer import NTNTrainer
from seq2seq.evaluator import NTNPredictor

from event_tensors.glove_utils import Glove
from model import NeuralTensorNetwork
from dataset_for_intent_prediction import Vocab, load_intent_prediction_dataset


def init_model(model):
    for param in model.parameters():
        param.data.uniform_(-0.08, 0.08)


def main(option):
    random.seed(option.random_seed)
    torch.manual_seed(option.random_seed)

    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level='INFO', stream=sys.stdout)

    glove = Glove(option.emb_file)
    logging.info('loaded embeddings from ' + option.emb_file)

    src_vocab = Vocab.build_from_glove(glove)
    tgt_vocab = Vocab.load(option.intent_vocab)

    train_dataset = load_intent_prediction_dataset(option.train_dataset, src_vocab, tgt_vocab, device=option.device)
    dev_dataset = load_intent_prediction_dataset(option.dev_dataset, src_vocab, tgt_vocab, device=option.device)

    train_data_loader = DataLoader(train_dataset, batch_size=option.batch_size, shuffle=True)
    dev_data_loader = DataLoader(dev_dataset, batch_size=len(dev_dataset), shuffle=False)

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    
    # Prepare loss
    weight = torch.ones(tgt_vocab_size)
    pad = tgt_vocab.stoi[tgt_vocab.pad_token]
    loss = Perplexity(weight, pad)
    loss.criterion.to(option.device)

    # Initialize model
    encoder = NeuralTensorNetwork(nn.Embedding(src_vocab_size, option.emb_dim), option.em_k)
    decoder = DecoderRNN(tgt_vocab_size, option.im_max_len, option.im_hidden_size,
                        use_attention=False, bidirectional=False,
                        eos_id=tgt_vocab.stoi[tgt_vocab.eos_token], sos_id=tgt_vocab.stoi[tgt_vocab.bos_token])
    encoder.to(option.device)
    decoder.to(option.device)

    init_model(encoder)
    init_model(decoder)

    encoder.embeddings.weight.data.copy_(torch.from_numpy(glove.embd).float())

    optimizer_params = [
        { 'params': encoder.parameters() },
        { 'params': decoder.parameters() }
    ]
    optimizer = Optimizer(optim.Adam(optimizer_params, lr=option.lr), max_grad_norm=5)
    trainer = NTNTrainer(loss, print_every=option.report_every, device=option.device)
    encoder, decoder = trainer.train(encoder, decoder, optimizer, train_data_loader, num_epochs=option.epochs, dev_data_loader=dev_data_loader, teacher_forcing_ratio=option.im_teacher_forcing_ratio)

    predictor = NTNPredictor(encoder, decoder, src_vocab, tgt_vocab, option.device)
    samples = [
        ("PersonX", "eventually told", "___"),
        ("PersonX", "tells", "PersonY 's tale"),
        ("PersonX", "always played", " ___"),
        ("PersonX", "would teach", "PersonY"),
        ("PersonX", "gets", "a ride"),
    ]
    for sample in samples:
        subj, verb, obj = sample
        subj = subj.lower().split(' ')
        verb = verb.lower().split(' ')
        obj = obj.lower().split(' ')
        print(sample, predictor.predict(subj, verb, obj))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=19960125)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--emb_file', type=str, default='/users5/kliao/code/CommonsenseERL_EMNLP_2019/data/glove.6B.100d.ext.txt')
    parser.add_argument('--train_dataset', type=str, default='/users5/kliao/code/CommonsenseERL_EMNLP_2019/data/atomic/train2.txt')
    parser.add_argument('--dev_dataset', type=str, default='/users5/kliao/code/CommonsenseERL_EMNLP_2019/data/atomic/dev2.txt')
    parser.add_argument('--intent_vocab', type=str, default='data/atomic/intent_vocab_train.json')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--emb_dim', type=int, default=100)
    parser.add_argument('--em_k', type=int, default=100)
    parser.add_argument('--im_max_len', type=int, default=10)
    parser.add_argument('--im_teacher_forcing_ratio', type=float, default=0.5)
    parser.add_argument('--im_hidden_size', type=int, default=100)
    parser.add_argument('--report_every', type=int, default=10)
    option = parser.parse_args()

    main(option)
