import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, v_measure_score


class ValidateAutoencoder:
    def __init__(self, dataloader, loss_func, device):
        self.device = device
        self.loss_func = loss_func
        self.dataloader = dataloader
        self.tot_test_loss = 0

    def test(self, model):
        model = model.to(self.device).double()
        self.tot_test_loss = 0
        for data_batch in self.dataloader:
            # data_batch = data_batch.double()
            data_batch = data_batch.to(self.device)
            output = model(data_batch)
            loss = self.loss_func(output, data_batch)
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
            # data_batch = data_batch.double()
            data_batch = data_batch.to(self.device)
            output = model(data_batch)
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
            colour_list = cm.rainbow(np.linspace(0,1,num_cluster))
            for point_idx_list, colour in zip(label_point_idx_map.values(), colour_list):
                plt.scatter(tsne_val[point_idx_list, 0], tsne_val[point_idx_list, 1], c=[colour,])
        else:
            plt.scatter(tsne_val[:, 0], tsne_val[:, 1])
        plt.savefig(fname=f'./figures/tnse.png')
        plt.show()


class EvaluateClustering:
    def __init__(self, Clustering, all_latent_vec, true_labels):
        self.Clustering = Clustering
        self.clustering_obj = None
        self.all_latent_vec = all_latent_vec
        self.sil_score = None
        self.sil_score_list = []
        self.vmeasure_score = None
        self.vmeasure_score_list = []
        self.distortion = None
        self.distortion_list = []
        self.true_labels = true_labels
        self._create_dir()

    def _create_dir(self):
        dir_list = os.listdir('./')
        if 'figures' not in dir_list:
            os.mkdir('figures')

    def _cal_fitted_cluster_model(self, no_cluster):
        self.clustering_obj = self.Clustering(no_cluster=no_cluster)
        self.clustering_obj.clustering_fit(all_latent_vec=self.all_latent_vec)
        return self.clustering_obj.get_pred_labels()

    def cal_sil_score(self, no_cluster):
        pred_labels = self._cal_fitted_cluster_model(no_cluster=no_cluster)
        self.sil_score = silhouette_score(self.all_latent_vec, pred_labels)

    def cal_vmeasure_score(self, no_cluster):
        pred_labels = self._cal_fitted_cluster_model(no_cluster=no_cluster)
        self.vmeasure_score = v_measure_score(labels_true=self.true_labels, labels_pred=pred_labels)

    def cal_sil_score_range(self, start, end, step):
        self.sil_score_list = []
        for no_cluster in range(start, end + 1, step):
            self.cal_sil_score(no_cluster=no_cluster)
            self.sil_score_list.append((no_cluster, self.get_sil_score()))

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

    def get_sil_score(self):
        return self.sil_score

    def get_sil_score_list(self):
        return self.sil_score_list

    def get_vmeasure_score(self):
        return self.vmeasure_score


class LabelCorrection:
    def __init__(self, decoder, maxpool_indices_array, device):
        self.device = device
        self.maxpool_indices_array = maxpool_indices_array.to(self.device)
        self.decoder = decoder.to(self.device).double()
        self.unflatten = torch.nn.Unflatten(dim=0, unflattened_size=(1, 64))
        self.label_map = {}

    def dis_cluster_centroid(self, all_latent_vec, cluster_centroid_idx):
        self.label_map = {}
        all_latent_vec = torch.tensor(all_latent_vec, dtype=torch.float64)
        for cluster_label, centroid_idx in cluster_centroid_idx.items():
            centroid_features = all_latent_vec[centroid_idx]
            decoder_input = self.unflatten(centroid_features)
            maxpool_idx = self.maxpool_indices_array[centroid_idx].reshape(1, 20, 12, 12)
            img_tensor = self.decoder(decoder_input.to(self.device),
                                      maxpool_idx)
            img_tensor = img_tensor[0].cpu()
            img_tensor = torch.transpose(img_tensor, 0, 2)
            img_tensor = torch.transpose(img_tensor, 0, 1)
            img_plt = img_tensor.detach().numpy()
            plt.close()
            plt.imshow(img_plt)
            plt.show()
            true_label = int(input('Enter the label (0-9): '))
            self.label_map[true_label] = cluster_label

    def get_label_map(self):
        return self.label_map
