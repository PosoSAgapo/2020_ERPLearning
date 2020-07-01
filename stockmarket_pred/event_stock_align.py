import torch
import argparse
from datetime import datetime, timedelta
import pickle

PATTERN = '%Y-%m-%d'


class StockData:
    def __init__(self, str_date, open, high, low, close, volume, adj_close):
        self.str_date = str_date
        self.date = datetime.strptime(str_date, PATTERN)
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.adj_close = adj_close

    def __str__(self):
        return '%s,%.2f,%.2f,%.2f,%.2f,%d,%.2f' % (self.str_date, self.open, self.high, self.low, self.close, self.volume, self.adj_close)


def load_stock_data_sp500(filename):
    stock_data_dict = {}
    lines = open(filename, 'r').readlines()[1:]
    for line in lines:
        line = line.strip().split(',')
        stock_data = StockData(line[0], float(line[1]), float(line[2]), float(line[3]), float(line[4]), int(line[5]), float(line[6]))
        stock_data_dict[stock_data.str_date] = stock_data
    return stock_data_dict


def align(stock_data_dict, event_data, start_date, end_date,true_event=None):
    pattern = '%Y-%m-%d'
    start_date = datetime.strptime(start_date, pattern)
    end_date = datetime.strptime(end_date, pattern) + timedelta(days=1)
    date = start_date - timedelta(days=1)
    all_labels = []
    all_events = []
    dates = []
    bge=[]
    while date < end_date:
        date += timedelta(days=1)
        str_date = datetime.strftime(date, pattern)
        if str_date not in stock_data_dict:
            continue

        dates.append(str_date)
        stock_data = stock_data_dict[str_date]
        label = 1 if stock_data.close > stock_data.open else 0
        all_labels.append(label)
        month_before = date - timedelta(days=30)
        
        events = []
        word_event = []
        for i in range(30):
            date2 = month_before + timedelta(days=i)
            str_date2 = datetime.strftime(date2, pattern)
            if str_date2 in event_data:
                print(str_date2)
                event = event_data[str_date2]
                #words_e = true_event[str_date2]
                try:
                    word_event.append(true_event[str_date2])
                except KeyError:
                    word_event.append('Key not found')
            else:
                event = torch.zeros(100).float()
            events.append(event)
        events = torch.stack(events)
        bge.append(word_event)
        all_events.append(events)

    all_labels = torch.LongTensor(all_labels)
    all_events = torch.stack(all_events)
    return all_labels, all_events, dates,bge


# def align(stock_data_dict, event_data, start_date, end_date):
#     pattern = '%Y-%m-%d'
#     start_date = datetime.strptime(start_date, pattern)
#     end_date = datetime.strptime(end_date, pattern)
#     date = start_date - timedelta(days=1)
#     all_labels = []
#     all_events = []
#     dates = []
#     while date < end_date:
#         date += timedelta(days=1)
#         str_date = datetime.strftime(date, pattern)
#         if str_date not in stock_data_dict:
#             continue
#         str_date_before = datetime.strftime(date - timedelta(days=1), pattern)
#         if str_date_before not in stock_data_dict:
#             str_date_before = datetime.strftime(date - timedelta(days=2), pattern)
#         if str_date_before not in stock_data_dict:
#             str_date_before = datetime.strftime(date - timedelta(days=3), pattern)
#         if str_date_before not in stock_data_dict:
#             str_date_before = datetime.strftime(date - timedelta(days=4), pattern)
#         if str_date_before not in stock_data_dict:
#             str_date_before = datetime.strftime(date - timedelta(days=5), pattern)
#         if str_date_before not in stock_data_dict:
#             str_date_before = datetime.strftime(date - timedelta(days=6), pattern)
#         if str_date_before not in stock_data_dict:
#             print(str_date, str_date_before)
#             continue

#         dates.append(str_date)

#         label = 1 if (stock_data_dict[str_date] > stock_data_dict[str_date_before]) else 0
#         all_labels.append(label)

#         month_before = date - timedelta(days=30)
#         events = []
#         for i in range(30):
#             date2 = month_before + timedelta(days=i)
#             str_date2 = datetime.strftime(date2, pattern)
#             if str_date2 in event_data:
#                 event = event_data[str_date2]
#             else:
#                 event = torch.zeros(100).float()
#             events.append(event)
#         events = torch.stack(events)
#         all_events.append(events)

#     all_labels = torch.LongTensor(all_labels)
#     all_events = torch.stack(all_events)
#     return all_labels, all_events, dates


def train_dev_test_split(label_data, event_data, dates, train_dev_split, dev_test_split,word_event):
    pattern = '%Y-%m-%d'
    train_dev_split = datetime.strptime(train_dev_split, pattern)
    dev_test_split = datetime.strptime(dev_test_split, pattern)

    train_labels = []
    train_events = []
    train_dates = []
    dev_labels = []
    dev_events = []
    dev_dates = []
    test_labels = []
    test_events = []
    test_dates = []
    test_word=[]

    for i in range(len(dates)):
        date = datetime.strptime(dates[i], pattern)
        if date < train_dev_split:
            train_labels.append(label_data[i])
            train_events.append(event_data[i])
            train_dates.append(dates[i])
        elif date < dev_test_split:
            dev_labels.append(label_data[i])
            dev_events.append(event_data[i])
            dev_dates.append(dates[i])
        else:
            test_labels.append(label_data[i])
            test_events.append(event_data[i])
            test_dates.append(dates[i])
            test_word.append(word_event[i])

    train_labels = torch.stack(train_labels)
    train_events = torch.stack(train_events)
    train_data = (train_labels, train_events, train_dates)
    dev_labels = torch.stack(dev_labels)
    dev_events = torch.stack(dev_events)
    dev_data = (dev_labels, dev_events, dev_dates)
    test_labels = torch.stack(test_labels)
    test_events = torch.stack(test_events)
    test_data = (test_labels, test_events, test_dates)
    return train_data, dev_data, test_data,test_word


def main1():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock_file', type=str, default='data/S&P500.csv')
    parser.add_argument('--event_file', type=str, default='data/emb/reverb.pt')
    parser.add_argument('--train_file', type=str, default='data/event/reverb_train.pt1')
    parser.add_argument('--dev_file', type=str, default='data/event/reverb_dev.pt1')
    parser.add_argument('--test_file', type=str, default='data/event/reverb_test.pt1')
    option = parser.parse_args()

    stock_file = option.stock_file
    event_file = option.event_file
    train_file = option.train_file
    dev_file = option.dev_file
    test_file = option.test_file
    true_event_file ='/users4/bwchen/CommonsenseERL_EMNLP_2019/stockmarket_pred/event_data' 

    stock_data_dict = load_stock_data_sp500(stock_file)
    event_data = torch.load(event_file)
    true_event = pickle.load(open(true_event_file,'rb'))

    labels, events, dates,word_event = align(stock_data_dict, event_data, '2006-10-02', '2013-11-21',true_event)
    train_data, dev_data, test_data,test_word = train_dev_test_split(labels, events, dates, '2012-06-19', '2013-02-22',word_event)

    print('train:', train_data[1].size())
    print('dev:', dev_data[1].size())
    print('test:', test_data[1].size())
    print(len(test_word))
    print(test_word)
    f=open('test_data','wb')
    pickle.dump(test_word,f)
    f.close()
    # torch.save(train_data, train_file)
    # torch.save(dev_data, dev_file)
    # torch.save(test_data, test_file)


if __name__ == '__main__':
    main1()