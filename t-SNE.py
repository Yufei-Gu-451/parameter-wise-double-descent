import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def visualize_scatter(data_2d, label_ids, figsize=(8, 6)):
    plt.figure(figsize=figsize)
    plt.grid()

    nb_classes = len(np.unique(label_ids))

    for label_id in np.unique(label_ids):
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    marker='o',
                    color=plt.cm.Set1(label_id / float(nb_classes)),
                    linewidth=1,
                    alpha=0.8,
                    label=id_to_label_dict[label_id])
    plt.legend(loc='best')
    plt.style.use('bmh')
    plt.title('t-SNE')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig('sarsim_sample.jpg')

# set dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.0,), (1.0,)),
])

dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

images = np.array(dataset.data)
labels = np.array(dataset.targets)

images_2d = images.reshape(-1, images.shape[-1])
images_scaled = StandardScaler().fit_transform(images_2d)

label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
label_ids = np.array([label_to_id_dict[x] for x in labels])

pca = PCA()
pca_result = pca.fit_transform(images_scaled)

tsne = TSNE(n_components=2, perplexity=40.0)
tsne_result = tsne.fit_transform(pca_result)
tsne_result_scaled = StandardScaler().fit_transform(tsne_result)
#tsne_result_scaled = tsne_result
visualize_scatter(tsne_result_scaled, label_ids)
