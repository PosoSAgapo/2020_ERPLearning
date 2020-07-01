def process(raw_file, event_file, output_file):
    d = {}
    data = []
    with open(raw_file, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split(' | ')
            data.append(line)
            d[line[1].replace(' ', '')] = i
    
    with open(event_file, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            event = '|'.join([line[2], line[3], line[4]])
            title = line[12].replace(' ', '')[:-1]
            if title not in d:
                continue
            i = d[title]
            data[i].append(event)

    with open(output_file, 'w') as f:
        for item in data:
            # if len(item) > 3:
            if True:
                f.write(' || '.join(item) + '\n')


def main():
    process('data/cooked/titles_train.txt', 'temp/reverb_train.txt', 'data/cooked/reverb_train.txt')
    process('data/cooked/titles_dev.txt', 'temp/reverb_dev.txt', 'data/cooked/reverb_dev.txt')
    process('data/cooked/titles_test.txt', 'temp/reverb_test.txt', 'data/cooked/reverb_test.txt')


if __name__ == '__main__':
    main()
