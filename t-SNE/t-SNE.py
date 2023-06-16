import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.0,), (1.0,)),
])

label_noise_ratio = 0.2
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

label_noise_transform = transforms.Lambda(lambda y: torch.tensor(np.random.randint(0, 10)))
num_samples = len(dataset)
num_noisy_samples = int(label_noise_ratio * num_samples)

noisy_indices = np.random.choice(num_samples, num_noisy_samples, replace=False)
for idx in noisy_indices:
    dataset.targets[idx] = label_noise_transform(dataset.targets[idx])

images = dataset.data
images = images.view(images.shape[0], -1).numpy()
labels = np.array(dataset.targets)

#pca = PCA(n_components=20)
#pca_result = pca.fit_transform(images)

# Instantiate and fit t-SNE on the data
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(images)

# Plot the t-SNE visualization
plt.figure(figsize=(30, 20))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.title('t-SNE Visualization of MNIST')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.savefig('t-SNE Visualization.jpg')
