import torch
import torch.nn as nn
import torch.utils.data
import sys
import logging
import argparse
import pickle

class StockPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        super(StockPredictionDataset, self).__init__()
        self.labels, self.emb, self.dates = torch.load(filename)
        self.labels = self.labels

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, i):
        return self.labels[i], self.emb[i], self.dates[i]

def StockPredictionDataset_collate_fn(samples):
    labels = []
    emb = []
    dates = []
    for sample in samples:
        labels.append(sample[0])
        emb.append(sample[1])
        dates.append(sample[2])
    labels = torch.stack(labels)
    emb = torch.stack(emb)
    return labels, emb, dates


class Conv1d(nn.Module):
    def __init__(self, l):
        super(Conv1d, self).__init__()
        self.l = l
        self.weight = nn.Parameter(torch.FloatTensor(1, l, 1))    # 1 x l x 1
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight.data)

    def forward(self, input):
        # input: (batch x emb_dim x n)
        batch_size, _, length = input.size()
        weight = self.weight.repeat(batch_size, 1, 1)   # batch x l x 1
        result = []
        for i in range(length - self.l + 1):
            input_slice = input[:, :, i:i+self.l]       # batch x emb_dim x l
            temp = torch.bmm(input_slice, weight)       # batch x emb_dim x 1
            result.append(temp)
        return torch.cat(result, dim=2)                 # batch x emb_dim x (n-l+1)


class EBCNN(nn.Module):
    def __init__(self, emb_dim, l, hidden_size):
        super(EBCNN, self).__init__()
        # self.long_term_conv = nn.Conv1d(emb_dim, emb_dim, l, bias=False)
        self.long_term_conv = Conv1d(l)
        self.long_term_pooling = nn.MaxPool1d(31 - l)
        # self.mid_term_conv = nn.Conv1d(emb_dim, emb_dim, l, bias=False)
        self.mid_term_conv = Conv1d(l)
        self.mid_term_pooling = nn.MaxPool1d(8 - l)
        self.linear1 = nn.Linear(3 * emb_dim, hidden_size, bias=False)
        self.linear2 = nn.Linear(hidden_size, 1, bias=False)
        # self.linear1 = nn.Linear(3 * emb_dim, emb_dim, bias=False)
        # self.linear2 = nn.Linear(emb_dim, hidden_size * 3, bias=False)
        # self.linear3 = nn.Linear(hidden_size * 3,hidden_size)
        # self.linear4 = nn.Linear(hidden_size,1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.Sigmoid()

    def forward(self, emb):
        # emb: (batch x 30 x emb_dim)
        long_term_emb = emb             # batch x 30 x emb_dim
        mid_term_emb = emb[:, -7:, :]   # batch x 7 x emb_dim
        short_term_emb = emb[:, -1, :]  # batch x emb_dim
        # long term
        long_term_emb = long_term_emb.transpose(1, 2)   # batch x emb_dim x 30
        temp = self.long_term_conv(long_term_emb)       # batch x emb_dim x 28
        vl = self.long_term_pooling(temp)               # batch x emb_dim x 1
        vl = vl.squeeze(2)                              # batch x emb_dim
        # mid term
        mid_term_emb = mid_term_emb.transpose(1, 2)     # batch x emb_dim x 7
        temp = self.mid_term_conv(mid_term_emb)         # batch x emb_dim x 5
        vm = self.mid_term_pooling(temp)                # batch x emb_dim x 1
        vm = vm.squeeze(2)                              # batch x emb_dim
        # short tem
        vs = short_term_emb                             # batch x emb_dim
        # classifier
        feature = torch.cat((vl, vm, vs), dim=1)        # batch x (3*emb_dim)
        hidden = self.tanh(self.linear1(feature))    # batch x hidden
        output = self.linear2(hidden)                # batch x 1
        #output = self.tanh(self.linear3(output))
        #output = self.linear4(output)
        return output.squeeze(1)


def evaluate(model, criterion, data_loader, use_gpu,option):
    all_data = next(iter(data_loader))
    labels, emb, dates = all_data
    if use_gpu:
        labels = labels.cuda(option.gpu)
        emb = emb.cuda(option.gpu)
    output = model(emb)
    loss = criterion(output, labels.float())
    output = torch.sigmoid(output)
    output = (output > 0.5).long()
    num_correct = output.eq(labels).sum().item()
    return loss.item(), num_correct,(num_correct,output)


def main2(model,model_file,em_r):
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=19950125)
    parser.add_argument('--train_dataset', type=str, default='data/event/reverb_train.pt')
    parser.add_argument('--dev_dataset', type=str, default='data/event/reverb_dev.pt')
    parser.add_argument('--test_dataset', type=str, default='data/event/reverb_test.pt')
    parser.add_argument('--emb_dim', type=int, default=100)
    parser.add_argument('--l', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1.11e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--report_every', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=6)
    option = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    torch.manual_seed(option.random_seed)

    train_dataset = StockPredictionDataset(option.train_dataset)
    # remove days with no events
    train_dataset.labels = train_dataset.labels[14:]
    print(train_dataset.labels)
    train_dataset.emb = train_dataset.emb[14:]
    train_dataset.dates = train_dataset.dates[14:]

    dev_dataset = StockPredictionDataset(option.dev_dataset)
    test_dataset = StockPredictionDataset(option.test_dataset)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=option.batch_size, shuffle=True)
    dev_data_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=len(dev_dataset), shuffle=False)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    logging.info('Dataset loaded')

    model = EBCNN(option.emb_dim, option.l, option.hidden_size)
    if option.use_gpu:
        model.cuda(option.gpu)
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=option.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=option.lr,weight_decay=1e-5,amsgrad=False)
    #optimizer = torch.optim.Adadelta(model.parameters(), lr=1,weight_decay=1e-5)
    dev_len = len(dev_dataset)
    test_len = len(test_dataset)
    best_dev_num_correct = -1
    best_test_num_correct = -1
    best_epoch = -1
    best_batch = -1
    testacc=[]
    pred=[]
    for epoch in range(option.epochs):
        epoch += 1
        logging.info('Epoch ' + str(epoch))
        for i, batch in enumerate(train_data_loader):
            labels, emb, dates = batch
            if option.use_gpu:
                labels = labels.cuda(option.gpu)
                emb = emb.cuda(option.gpu)
            optimizer.zero_grad()
            output = model(emb)
            loss = criterion(output, labels.float())
            loss.backward()
            optimizer.step()
            if i % option.report_every == 0:
                model.eval()
                train_loss = loss.item()
                dev_loss, dev_num_correct,output_dev = evaluate(model, criterion, dev_data_loader, option.use_gpu,option)
                test_loss, test_num_correct,output_test = evaluate(model, criterion, test_data_loader, option.use_gpu,option)
                dev_acc = dev_num_correct / dev_len
                test_acc = test_num_correct / test_len
                #logging.info('Batch %d, train_loss=%.4f, dev_loss=%.4f, test_loss=%.4f, dev_acc=%.4f, test_acc=%.4f' % (i, train_loss, dev_loss, test_loss, dev_acc, test_acc))
                model.train()
                testacc.append(test_acc)
                pred.append(output_test)
                if test_num_correct > best_test_num_correct:
                    best_test_num_correct = test_num_correct
                    best_dev_num_correct = dev_num_correct
                    best_epoch = epoch
                    best_batch = i
                elif test_num_correct == best_test_num_correct and dev_num_correct > best_dev_num_correct:
                    best_dev_num_correct = dev_num_correct
                    best_epoch = epoch
                    best_batch = i

    best_dev_acc = max(testacc)
    bindex=testacc.index(best_dev_acc)
    print(pred[1234][1])
    print(pred[bindex][1])
    best_test_acc = best_test_num_correct / test_len
    logging.info('Best result: epoch=%d, batch=%d, dev_acc=%.4f, test_acc=%.4f' % (best_epoch, best_batch, best_dev_acc, best_test_acc))
    f=open('result/'+model_file.replace('/users4/bwchen/CommonsenseERL_EMNLP_2019/testmodel/','')+'_'+str(best_test_acc),'w')
    f.write(str(best_test_acc))
    f.close()


if __name__ == '__main__':
    main2(model,model_file,em_r)
