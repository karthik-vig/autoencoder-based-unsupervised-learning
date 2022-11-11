import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Func
from sklearn.cluster import KMeans


# encoder:
# autoencoder model:

# encoder:
class ConvEncoder(nn.Module):
    def __init__(self, latent_dims, encoder_setup):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=encoder_setup['conv1']['in_ch'],
                               out_channels=encoder_setup['conv1']['out_ch'],
                               kernel_size=encoder_setup['conv1']['kernel_sz'],
                               stride=encoder_setup['conv1']['stride'], padding=0)
        self.conv2 = nn.Conv2d(in_channels=encoder_setup['conv2']['in_ch'],
                               out_channels=encoder_setup['conv2']['out_ch'],
                               kernel_size=encoder_setup['conv2']['kernel_sz'],
                               stride=encoder_setup['conv2']['stride'], padding=0)
        self.conv3 = nn.Conv2d(in_channels=encoder_setup['conv3']['in_ch'],
                               out_channels=encoder_setup['conv3']['out_ch'],
                               kernel_size=encoder_setup['conv3']['kernel_sz'],
                               stride=encoder_setup['conv3']['stride'], padding=0)
        self.maxPool = nn.MaxPool2d(kernel_size=encoder_setup['pool']['kernel_sz'],
                                    stride=encoder_setup['pool']['stride'], return_indices=True)
        self.linear1 = nn.Linear(in_features=encoder_setup['ln1']['in_feat'],
                                 out_features=encoder_setup['ln1']['out_feat'])
        self.linear2 = nn.Linear(in_features=encoder_setup['ln2']['in_feat'],
                                 out_features=encoder_setup['ln2']['out_feat'])
        self.linear3 = nn.Linear(in_features=encoder_setup['ln3']['in_feat'],
                                 out_features=encoder_setup['ln3']['out_feat'])
        self.linear4 = nn.Linear(in_features=encoder_setup['ln4']['in_feat'],
                                 out_features=encoder_setup['ln4']['out_feat'])
        self.mean_linear = nn.Linear(in_features=encoder_setup['mean_ln']['in_feat'], out_features=latent_dims)
        self.std_linear = nn.Linear(in_features=encoder_setup['std_ln']['in_feat'], out_features=latent_dims)
        self.maxpool_indices = None
        # self.mean_x = None
        # self.std_x = None
        self.norm_dist = torch.distributions.Normal(0, 1)

    def forward(self, x):
        x = self.conv1(x)
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
        x = Func.relu(x)
        mean_x = self.mean_linear(x)
        std_x = torch.exp(self.std_linear(x))
        x = mean_x + std_x * self.norm_dist.sample()
        return x, mean_x, std_x

    def get_maxpool_indices(self):
        return self.maxpool_indices

    # def get_mean_std(self):
    #     return self.mean_x, self.std_x


# decoder:
class ConvDecoder(nn.Module):
    def __init__(self, latent_dims, decoder_setup):
        super(ConvDecoder, self).__init__()
        self.linear1 = nn.Linear(in_features=latent_dims, out_features=decoder_setup['ln1']['out_feat'])
        self.linear2 = nn.Linear(in_features=decoder_setup['ln2']['in_feat'],
                                 out_features=decoder_setup['ln2']['out_feat'])
        self.linear3 = nn.Linear(in_features=decoder_setup['ln3']['in_feat'],
                                 out_features=decoder_setup['ln3']['out_feat'])
        self.linear4 = nn.Linear(in_features=decoder_setup['ln4']['in_feat'],
                                 out_features=decoder_setup['ln4']['out_feat'])
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=decoder_setup['unflatten']['sz'])
        self.invMaxPool = nn.MaxUnpool2d(kernel_size=decoder_setup['invpool']['kernel_sz'],
                                         stride=decoder_setup['invpool']['stride'], padding=0)
        self.deconv1 = nn.ConvTranspose2d(in_channels=decoder_setup['deconv1']['in_ch'],
                                          out_channels=decoder_setup['deconv1']['out_ch'],
                                          kernel_size=decoder_setup['deconv1']['kernel_sz'],
                                          stride=decoder_setup['deconv1']['stride'], padding=0, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=decoder_setup['deconv2']['in_ch'],
                                          out_channels=decoder_setup['deconv2']['out_ch'],
                                          kernel_size=decoder_setup['deconv2']['kernel_sz'],
                                          stride=decoder_setup['deconv2']['stride'], padding=0, output_padding=0)
        self.deconv3 = nn.ConvTranspose2d(in_channels=decoder_setup['deconv3']['in_ch'],
                                          out_channels=decoder_setup['deconv3']['out_ch'],
                                          kernel_size=decoder_setup['deconv3']['kernel_sz'],
                                          stride=decoder_setup['deconv3']['stride'], padding=0, output_padding=0)

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
class VAE(nn.Module):
    def __init__(self, latent_dims, autoencoder_setup):
        super(VAE, self).__init__()
        self.encoder = ConvEncoder(latent_dims, autoencoder_setup['encoder'])
        self.decoder = ConvDecoder(latent_dims, autoencoder_setup['decoder'])
        # self.latent_vec = None
        # self.kl_div_val = None
        # self.mse_loss_val = None
        # self.loss_val = None
        self.log_std = nn.Parameter(torch.tensor([0.0]))

    def likelihood(self, x_hat, x):
        std = torch.exp(self.log_std)
        p = torch.distributions.Normal(x_hat, std)
        log_p = p.log_prob(x).sum(dim=(1, 2, 3))
        return log_p

    # def mse_loss(self, x_hat, x):
    #     return torch.square((x_hat - x)).sum(dim=(1, 2, 3))

    def kl_div(self, x, mean, sigma):
        p = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(sigma))
        q = torch.distributions.Normal(mean, sigma)
        log_q = q.log_prob(x)
        log_p = p.log_prob(x)
        kl_div_val = (log_q - log_p).sum(-1)
        return kl_div_val

    def forward(self, input):
        latent_vector, mean_x, std_x = self.encoder(input)
        maxpool_indices = self.encoder.get_maxpool_indices()
        # mean_x, std_x = self.encoder.get_mean_std()
        kl_div_val = self.kl_div(x=latent_vector, mean=mean_x, sigma=std_x)
        x_hat = self.decoder(latent_vector, maxpool_indices)
        # self.mse_loss_val = self.mse_loss(x_hat=x_hat, x=input)
        recon_loss = self.likelihood(x_hat=x_hat, x=input)
        loss_val = (kl_div_val - recon_loss).sum()
        return x_hat, loss_val

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    # def get_latent_vec(self):
    #     return self.latent_vec

    # def get_loss(self):
    #     return self.loss_val

    # def get_kl_div_val(self):
    #     return self.kl_div_val
    #
    # def get_mse_loss_val(self):
    #     return self.mse_loss_val


# class VAELoss:


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
