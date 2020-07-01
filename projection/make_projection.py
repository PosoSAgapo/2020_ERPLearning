import torch
from torch.utils.tensorboard import SummaryWriter
import os
from argparse import ArgumentParser


def write_summary(data, output_dir):
    os.system('rm -rf ' + output_dir)
    os.system('mkdir ' + output_dir)

    writer = SummaryWriter(output_dir)
    writer.add_embedding(data['embeddings'], metadata=data['labels'])
    writer.close()

    # fix metadata manually
    metadata_file = os.path.join(output_dir, '00000/default/metadata.tsv')
    with open(metadata_file, 'w') as f:
        f.write('label\ttext\n')
        for label, event_text in zip(data['labels'], data['event_texts']):
            f.write(str(label.item()) + '\t' + event_text + '\n')


def main(option):
    good = torch.load(option.good_embedding, map_location='cpu')
    bad = torch.load(option.bad_embedding, map_location='cpu')

    write_summary(good, option.good_summary)
    write_summary(bad, option.bad_summary)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--good_embedding', type=str, default='good_embedding.pt')
    parser.add_argument('--bad_embedding', type=str, default='bad_embedding.pt')
    parser.add_argument('--good_summary', type=str, default='runs/good')
    parser.add_argument('--bad_summary', type=str, default='runs/bad')
    option = parser.parse_args()

    main(option)
