from glob import glob
from datetime import datetime
from common import *


def write_file(filename, data):
    with open(filename, 'w') as f:
        for item in data:
            f.write(' | '.join([item[0], item[1], datetime.strftime(item[2], '%Y-%m-%d')]) + '\n')


def split_bloomberg():
    train_data = []
    dev_data = []
    test_data = []

    bloomberg_dirs = glob('data/20061020_20131126_bloomberg_news/*')
    for dirname in bloomberg_dirs:
        date = datetime.strptime(dirname.split('/')[-1], '%Y-%m-%d')
        if TRAIN_START_TIME <= date < TRAIN_END_TIME:
            data = train_data
        elif DEV_START_TIME <= date < DEV_END_TIME:
            data = dev_data
        elif TEST_START_TIME <= date < TEST_END_TIME:
            data = test_data
        else:
            continue
        filenames = glob(dirname + '/*')
        for filename in filenames:
            with open(filename, 'r') as f:
                s = f.readline().strip()
                if s == '--':
                    title = f.readline().strip()
                else:
                    title = s[3:]
            data.append(('bloomberg', title, date))

    return train_data, dev_data, test_data


def split_reuters():
    def parse_time(s):
        # -- Fri Oct 20, 2006 6:15pm EDT
        s = s[7:].strip()
        return datetime.strptime(s, '%b %d, %Y %I:%M%p EDT')

    train_data = []
    dev_data = []
    test_data = []

    bloomberg_dirs = glob('data/ReutersNews106521/*')
    for dirname in bloomberg_dirs:
        date = datetime.strptime(dirname.split('/')[-1], '%Y%m%d')
        if TRAIN_START_TIME <= date < TRAIN_END_TIME:
            data = train_data
        elif DEV_START_TIME <= date < DEV_END_TIME:
            data = dev_data
        elif TEST_START_TIME <= date < TEST_END_TIME:
            data = test_data
        else:
            continue
        filenames = glob(dirname + '/*')
        for filename in filenames:
            with open(filename, 'r') as f:
                title = f.readline()[3:].strip()
            data.append(('reuters', title, date))

    return train_data, dev_data, test_data


def main():
    bloomberg_data = split_bloomberg()
    print('Bloomberg done')
    reuters_data = split_reuters()
    print('Reuters done')

    train_data = bloomberg_data[0] + reuters_data[0]
    dev_data = bloomberg_data[1] + reuters_data[1]
    test_data = bloomberg_data[2] + reuters_data[2]

    train_data = sorted(train_data, key=lambda x: x[2])
    dev_data = sorted(dev_data, key=lambda x: x[2])
    test_data = sorted(test_data, key=lambda x: x[2])

    write_file('data/cooked/titles_train.txt', train_data)
    write_file('data/cooked/titles_dev.txt', dev_data)
    write_file('data/cooked/titles_test.txt', test_data)


if __name__ == '__main__':
    main()
