import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Func
from sklearn.cluster  import KMeans


# autoencoder model:

# encoder:
# autoencoder model:

# encoder:
class ConvEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=30, kernel_size=3, stride=1, padding=0)
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.linear1 = nn.Linear(in_features=30 * 11 * 11, out_features=1024)
        self.linear2 = nn.Linear(in_features=1024, out_features=512)
        self.linear3 = nn.Linear(in_features=512, out_features=256)
        self.linear4 = nn.Linear(in_features=256, out_features=latent_dims)
        self.maxpool_indices = None

    def forward(self, input):
        x = self.conv1(input)
        x = Func.relu(x)
        x = self.conv2(x)
        x = Func.relu(x)
        x = self.conv3(x)
        x = Func.relu(x)
        x, self.maxpool_indices = self.maxPool(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = Func.relu(x)
        x = self.linear2(x)
        x = Func.relu(x)
        x = self.linear3(x)
        x = Func.relu(x)
        x = self.linear4(x)
        return x

    def get_maxpool_indices(self):
        return self.maxpool_indices


# decoder:
class ConvDecoder(nn.Module):
    def __init__(self, latent_dims):
        super(ConvDecoder, self).__init__()
        self.linear1 = nn.Linear(in_features=latent_dims, out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=512)
        self.linear3 = nn.Linear(in_features=512, out_features=1024)
        self.linear4 = nn.Linear(in_features=1024, out_features=30 * 11 * 11)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(30, 11, 11))
        self.invMaxPool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.deconv1 = nn.ConvTranspose2d(in_channels=30, out_channels=20, kernel_size=3, stride=1, padding=0, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=20, out_channels=10, kernel_size=3, stride=1, padding=0, output_padding=0)
        self.deconv3 = nn.ConvTranspose2d(in_channels=10, out_channels=3, kernel_size=3, stride=1, padding=0, output_padding=0)

    def forward(self, latent_vector, maxpool_indices):
        x = self.linear1(latent_vector)
        x = Func.relu(x)
        x = self.linear2(x)
        x = Func.relu(x)
        x = self.linear3(x)
        x = Func.relu(x)
        x = self.linear4(x)
        x = Func.relu(x)
        x = self.unflatten(x)
        x = self.invMaxPool(x, maxpool_indices)
        x = self.deconv1(x)
        x = Func.relu(x)
        x = self.deconv2(x)
        x = Func.relu(x)
        x = self.deconv3(x)
        x = torch.sigmoid(x)
        return x


# autoencoder:
class Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = ConvEncoder(latent_dims)
        self.decoder = ConvDecoder(latent_dims)

    def forward(self, input):
        latent_vector = self.encoder(input)
        maxpool_indices = self.encoder.get_maxpool_indices()
        return self.decoder(latent_vector, maxpool_indices)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder


class Clustering:
    def __init__(self, no_cluster):
        self.no_cluster = no_cluster
        self.labels = None
        self.fitted_cluster_model = None
        self.cluster_centroid_idx = {}

    def clustering_fit(self, all_latent_vec):
        cluster_obj = KMeans(n_clusters=self.no_cluster)
        self.fitted_cluster_model = cluster_obj.fit(all_latent_vec)

    def clustering_predict(self, all_latent_vec):
        self.labels = self.fitted_cluster_model.predict(all_latent_vec)

    def get_pred_labels(self):
        return self.labels

    def get_clustering_model(self):
        return self.fitted_cluster_model

    def get_distortion(self):
        if self.fitted_cluster_model != None:
            return self.fitted_cluster_model.inertia_
        else:
            return None

    def get_cluster_centers(self):
        if self.fitted_cluster_model != None:
            return self.fitted_cluster_model.cluster_centers_
        else:
            return None

    def cal_cluster_centroid(self, all_latent_vec):
        self.cluster_centroid_idx = {}
        cluster_centers = self.get_cluster_centers()
        for cluster_label, cluster_cen in enumerate(cluster_centers):
            distance = np.sqrt(np.sum(np.power(all_latent_vec - cluster_cen, 2), axis=1))
            min_dist_idx = np.where(distance == np.min(distance))[0][0]
            # print(f'min distance is: {np.min(distance)}')
            self.cluster_centroid_idx[cluster_label] = min_dist_idx

    def get_cluster_centroid(self):
        return self.cluster_centroid_idx



