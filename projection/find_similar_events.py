import torch
import faiss
from argparse import ArgumentParser


def main(option):
    data = torch.load(option.embeddings, map_location='cpu')
    embeddings = data['embeddings']
    labels = data['labels']
    event_texts = data['event_texts']

    event_text = '(' + option.event.lower().replace(' | ', ', ') + ')'
    if event_text not in event_texts:
        print('event not found in embeddings')
        exit(1)

    # normalize
    embeddings = embeddings / embeddings.pow(2).sum(dim=1).unsqueeze(1)
    embeddings = embeddings.numpy()

    query_index = event_texts.index(event_text)
    query = embeddings[query_index : query_index + 1]

    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)
    data, indices = faiss_index.search(query, option.num_neighbors)

    for index in indices[0]:
        print(event_texts[index][1:-1].replace(', ', ' | '))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--event', type=str, default='PersonX | eats | chicken')
    parser.add_argument('--embeddings', type=str, default='embedding.pt')
    parser.add_argument('--num_neighbors', type=int, default=500)
    option = parser.parse_args()

    main(option)
