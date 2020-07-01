def process_file(input_file, output_file):
    input_file = open(input_file, 'r')
    output_file = open(output_file, 'w')
    for line in input_file:
        line = line.strip()
        _, title, _ = line.split(' | ')
        output_file.write(title + '.\n')
    input_file.close()
    output_file.close()


def main():
    process_file('data/cooked/titles_train.txt', 'temp/titles_train.txt')
    process_file('data/cooked/titles_dev.txt', 'temp/titles_dev.txt')
    process_file('data/cooked/titles_test.txt', 'temp/titles_test.txt')


if __name__ == '__main__':
    main()
