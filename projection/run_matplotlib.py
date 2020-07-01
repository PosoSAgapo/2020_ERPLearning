import torch
import json
from matplotlib import pyplot as plt

data = json.load(open('good_state.json', 'r'))
projections = data[0]['projections']
data = torch.load('good_embedding.pt', map_location='cpu')
labels = data['labels'].numpy()

# data = json.load(open('bad_state.json', 'r'))
# projections = data[0]['projections']
# data = torch.load('bad_embedding.pt', map_location='cpu')
# labels = data['labels'].numpy()

def plot_pca_scatter(projections, labels, xmin=-1, xmax=1, ymin=-0.9, ymax=1.0):
    colors = ['#3976AF', '#C53833', '#D67DBE', '#57BCCC']
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    for projection, label in zip(projections, labels):
        plt.scatter(projection['pca-0'], projection['pca-1'], c=colors[label])
    # plt.legend(np.arange(0, 10).astype(str))
    # plt.xlabel('First Principal Component')
    # plt.ylabel('Second Principal Component')
    plt.axis('off')
    plt.show()

plot_pca_scatter(projections, labels)
