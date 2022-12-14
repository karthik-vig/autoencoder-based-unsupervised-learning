import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, v_measure_score


class ValidateAutoencoder:
    def __init__(self, dataloader, device):
        self.device = device
        self.dataloader = dataloader
        self.tot_test_loss = 0

    def test(self, model):
        model = model.to(self.device).double()
        self.tot_test_loss = 0
        for data_batch in self.dataloader:
            data_batch = data_batch.to(self.device)
            output, loss = model(data_batch)
            # loss = self.loss_func(output, data_batch)
            self.tot_test_loss += loss.clone().detach()

    def get_test_loss(self):
        return self.tot_test_loss


class PlotAutoencoderGraph:
    def __init__(self, device):
        self.device = device
        self._create_dir()

    def _create_dir(self):
        dir_list = os.listdir('./')
        if 'figures' not in dir_list:
            os.mkdir('figures')

    def plot_loss(self, no_epoch, loss_list, title):
        plt.close()
        plt.figure(figsize=(20, 15), dpi=100)
        plt.plot([i for i in range(1, no_epoch + 1, 1)], loss_list)
        plt.xticks(range(1, no_epoch + 1, 1))
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(fname=f'./figures/{title}_loss.png')
        plt.show()

    def _convert_to_plt_image(self, img_tensor):
        img_tensor = img_tensor.cpu()
        img_tensor = torch.transpose(img_tensor, 0, 2)
        img_tensor = torch.transpose(img_tensor, 0, 1)
        return img_tensor.numpy()

    def draw_image(self, dataloader, model):
        model = model.to(self.device).double()
        plt.close()
        fig, ax = plt.subplots(2, 5, figsize=(15, 8))
        for idx, data_batch in enumerate(dataloader):
            if idx == 5:
                break
            data_batch = data_batch.to(self.device)
            output, loss = model(data_batch)
            original_img = self._convert_to_plt_image(img_tensor=data_batch[0])
            recons_img = self._convert_to_plt_image(img_tensor=output[0].detach())
            ax[0, idx].imshow(original_img)
            ax[1, idx].imshow(recons_img)
        ax[0, 0].set_ylabel('Original Images')
        ax[1, 0].set_ylabel('Reconstructed Images')
        plt.savefig(fname=f'./figures/original_and_recons.png')
        plt.show()

    def draw_tsne(self, all_latent_vec, label_point_idx_map=None):
        tsne_obj = TSNE(n_components=2, perplexity=50)
        tsne_val = tsne_obj.fit_transform(all_latent_vec)
        plt.close()
        plt.figure(figsize=(20, 15), dpi=100)
        if label_point_idx_map:
            num_cluster = len(label_point_idx_map)
            colour_list = cm.rainbow(np.linspace(0, 1, num_cluster))
            for point_idx_list, colour in zip(label_point_idx_map.values(), colour_list):
                plt.scatter(tsne_val[point_idx_list, 0], tsne_val[point_idx_list, 1], c=[colour, ])
        else:
            plt.scatter(tsne_val[:, 0], tsne_val[:, 1])
        plt.savefig(fname=f'./figures/tnse.png')
        plt.show()


class EvaluateClustering:
    def __init__(self, Clustering):
        self.Clustering = Clustering
        self.clustering_obj = None
        self.all_latent_vec = None
        self.sil_score = None
        self.sil_score_list = []
        self.vmeasure_score = None
        self.vmeasure_score_list = []
        self.distortion = None
        self.distortion_list = []
        self._create_dir()

    def set_all_latent_vec(self, all_latent_vec):
        self.all_latent_vec = all_latent_vec

    def _create_dir(self):
        dir_list = os.listdir('./')
        if 'figures' not in dir_list:
            os.mkdir('figures')

    def _cal_fitted_cluster_model(self, no_cluster):
        self.clustering_obj = self.Clustering(no_cluster=no_cluster)
        self.clustering_obj.clustering_fit(all_latent_vec=self.all_latent_vec)

    def cal_distortion(self, no_cluster):
        self._cal_fitted_cluster_model(no_cluster=no_cluster)
        self.distortion = self.clustering_obj.get_distortion()

    def cal_sil_score(self, no_cluster):
        self._cal_fitted_cluster_model(no_cluster=no_cluster)
        self.clustering_obj.clustering_predict(all_latent_vec=self.all_latent_vec)
        pred_labels = self.clustering_obj.get_pred_labels()
        self.sil_score = silhouette_score(self.all_latent_vec, pred_labels)

    def cal_vmeasure_score(self, no_cluster, true_labels):
        self._cal_fitted_cluster_model(no_cluster=no_cluster)
        self.clustering_obj.clustering_predict(all_latent_vec=self.all_latent_vec)
        pred_labels = self.clustering_obj.get_pred_labels()
        self.vmeasure_score = v_measure_score(labels_true=true_labels, labels_pred=pred_labels)

    def cal_sil_score_range(self, start, end, step):
        self.sil_score_list = []
        for no_cluster in range(start, end + 1, step):
            self.cal_sil_score(no_cluster=no_cluster)
            self.sil_score_list.append((no_cluster, self.get_sil_score()))

    def cal_distortion_range(self, start, end, step):
        self.distortion_list = []
        for no_cluster in range(start, end + 1, step):
            self.cal_distortion(no_cluster=no_cluster)
            self.distortion_list.append((no_cluster, self.get_distortion()))

    def draw_sil_score_list(self):
        if len(self.sil_score_list) == 0:
            raise Exception('silhouette score not calculated')
        plt.close()
        plt.figure(figsize=(20, 15), dpi=100)
        sil_score_array = np.array(self.sil_score_list)
        plt.plot(sil_score_array[:, 0], sil_score_array[:, 1])
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.savefig(fname='./figures/sil_score.png')
        plt.show()

    def draw_distortion_list(self):
        plt.close()
        plt.figure(figsize=(20, 15), dpi=100)
        distortion_array = np.array(self.distortion_list)
        plt.plot(distortion_array[:, 0], distortion_array[:, 1])
        plt.xlabel('Number of Clusters')
        plt.ylabel('Distortion')
        plt.savefig(fname=f'./figures/distortion.png')
        plt.show()

    def get_sil_score(self):
        return self.sil_score

    def get_sil_score_list(self):
        return self.sil_score_list

    def get_vmeasure_score(self):
        return self.vmeasure_score

    def get_distortion(self):
        return self.distortion

    def get_distortion_list(self):
        return self.distortion_list


class LabelCorrection:
    def __init__(self, device, latent_dim, maxpool_size):
        self.device = device
        self.latent_dim = latent_dim
        self.maxpool_indices_array = None
        self.decoder = None
        self.unflatten = torch.nn.Unflatten(dim=0, unflattened_size=(1, self.latent_dim))
        self.label_map = {}
        self.maxpool_size = maxpool_size

    def set_decoder(self, decoder):
        self.decoder = decoder.to(self.device).double()

    def set_maxpool_indices(self, maxpool_indices_array):
        self.maxpool_indices_array = maxpool_indices_array.to(self.device)

    def dis_cluster_centroid(self, all_latent_vec, cluster_centroid_idx):
        self.label_map = {}
        all_latent_vec = torch.tensor(all_latent_vec, dtype=torch.float64)
        plt.ion()
        fig, ax = plt.subplots(1, len(cluster_centroid_idx), figsize=(15, 8))
        for idx, (cluster_label, centroid_idx) in enumerate(cluster_centroid_idx.items()):
            centroid_features = all_latent_vec[centroid_idx]
            decoder_input = self.unflatten(centroid_features)
            maxpool_idx = self.maxpool_indices_array[centroid_idx].reshape(self.maxpool_size)
            img_tensor = self.decoder(decoder_input.to(self.device),
                                      maxpool_idx)
            img_tensor = img_tensor[0].cpu()
            img_tensor = torch.transpose(img_tensor, 0, 2)
            img_tensor = torch.transpose(img_tensor, 0, 1)
            img_plt = img_tensor.detach().numpy()
            ax[idx].imshow(img_plt)
            plt.pause(0.002)
        plt.pause(2)
        plt.show()
        for cluster_label in cluster_centroid_idx.keys():
            true_label = int(input('Enter the label (0-9): '))
            self.label_map[true_label] = cluster_label
        plt.ioff()

    def get_label_map(self):
        return self.label_map