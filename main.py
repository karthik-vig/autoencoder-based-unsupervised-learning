import torch.nn as nn
import torch.optim as optim

from evaluation import ValidateAutoencoder, PlotAutoencoderGraph, EvaluateClustering, LabelCorrection
from ml_models import Autoencoder, Clustering
from process_data import LoadData, TrainAutoencoder, SaveLoadAutoencoderModel, LatentVecConversion


# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import numpy as np
# from joblib import Memory

class Execute:
    def __init__(self, **kwargs):
        self.autoencoder_model = Autoencoder(latent_dims=kwargs['latent_dim'])
        self.autoencoder_adam_optimizer = optim.Adam(self.autoencoder_model.parameters(), lr=kwargs['lr'])
        self.mse_loss_func = nn.MSELoss()

        self.load_data_obj = LoadData(batch_size=kwargs['batch_size'],
                                      image_dim=kwargs['img_dim'],
                                      test_dataset_loc=kwargs['test_dataset_loc'],
                                      train_dataset_loc=kwargs['train_dataset_loc'])
        self.load_data_obj.load_dataset()
        self.load_data_obj.create_dataloaders()

        self.train_autoencoder_obj = TrainAutoencoder(train_dataloader=self.load_data_obj.get_train_dataloader(),
                                                      model=self.autoencoder_model,
                                                      loss_func=self.mse_loss_func,
                                                      optimizer=self.autoencoder_adam_optimizer,
                                                      device=kwargs['device'])

        self.test_autoencoder_obj = ValidateAutoencoder(dataloader=self.load_data_obj.get_validation_dataloader(),
                                                        loss_func=self.mse_loss_func,
                                                        device=kwargs['device'])

        self.save_load_autoencoder_obj = SaveLoadAutoencoderModel(file_dir='autoencoder_models')

        self.plot_graph_obj = PlotAutoencoderGraph(device=kwargs['device'])

        self.clustering_obj = Clustering(no_cluster=kwargs['num_cluster'])

        self.transform_to_latent_vec_obj = LatentVecConversion(device='cuda', latent_dim=kwargs['latent_dim'])

        self.eva_cluster_obj = EvaluateClustering(Clustering=Clustering)

        self.label_correc_obj = LabelCorrection(device='cuda',
                                                latent_dim=64)

        self.latent_dim = kwargs['latent_dim']
        self.model_data = None
        # self.model = None
        # self.encoder = None
        # self.decoder = None
        self.all_latent_train_vec_model_num = None
        self.all_latent_train_vec = None
        self.train_maxpool_indices = None
        self.train_pred_labels = None

    def execute_train_autoencoder(self, start_epoch, end_epoch):
        for epoch in range(start_epoch, end_epoch + 1):
            # train model
            self.train_autoencoder_obj.train()
            model = self.train_autoencoder_obj.get_model()
            train_loss = self.train_autoencoder_obj.get_train_loss()
            # validate the model
            self.test_autoencoder_obj.test(model=model)
            test_loss = self.test_autoencoder_obj.get_test_loss()
            # create a structure to store model information as well as the meta-data associated with it
            model_name = f'autoencoder_model_{epoch}.pt'
            model_data = {'model': model,
                          'meta_data': {'train_loss': train_loss,
                                        'test_loss': test_loss}}
            # save the model information
            self.save_load_autoencoder_obj.save(model=model_data, model_name=model_name)
            # print the loss values for the model
            print(f'''The Train loss is: {train_loss}\r
            The test loss is: {test_loss}\r
            Epoch: {epoch} is done.\n''')

    def cal_latent_vec_for_model(self, select_model):
        if select_model != self.all_latent_train_vec_model_num:
            self.model_data = self.save_load_autoencoder_obj.load(select_model=select_model)
            model = self.model_data['model']
            self.transform_to_latent_vec_obj.set_model(model=model)
            self.transform_to_latent_vec_obj.cal_latent_vec(dataloader=self.load_data_obj.get_train_dataloader())
            self.all_latent_train_vec = self.transform_to_latent_vec_obj.get_all_latent_vec()  # use the cluster centers to find a point
            self.all_latent_train_vec_model_num = select_model
            self.train_maxpool_indices = self.transform_to_latent_vec_obj.get_maxpool_indices()

    def execute_train_cluster_model(self, select_model):
        self.cal_latent_vec_for_model(select_model=select_model)
        self.clustering_obj.clustering_fit(all_latent_vec=self.all_latent_train_vec)
        self.clustering_obj.clustering_predict(all_latent_vec=self.all_latent_train_vec)
        self.train_pred_labels = self.clustering_obj.get_pred_labels()

    def execute_eva_autoencoder(self, select_model):
        train_loss, test_loss, no_epochs = self.save_load_autoencoder_obj.get_model_data()

        model_data = self.save_load_autoencoder_obj.load(select_model=select_model)
        model = model_data['model']
        self.plot_graph_obj.plot_loss(no_epoch=no_epochs,
                                      loss_list=train_loss,
                                      title='Train Loss')
        self.plot_graph_obj.plot_loss(no_epoch=no_epochs,
                                      loss_list=test_loss,
                                      title='Test Loss')
        self.plot_graph_obj.draw_image(dataloader=self.load_data_obj.get_validation_dataloader(),
                                       model=model)

    def cal_sil(self, select_model, start, end, step):
        self.cal_latent_vec_for_model(select_model=select_model)
        self.eva_cluster_obj.set_all_latent_vec(all_latent_vec=self.all_latent_train_vec)
        self.eva_cluster_obj.cal_sil_score_range(start=start, end=end, step=step)
        sil_score_list = self.eva_cluster_obj.get_sil_score_list()
        print(f'The silhouette score list is: {sil_score_list}')
        self.eva_cluster_obj.draw_sil_score_list()

    def cal_dis(self, select_model, start, end, step):
        self.cal_latent_vec_for_model(select_model=select_model)
        self.eva_cluster_obj.set_all_latent_vec(all_latent_vec=self.all_latent_train_vec)
        self.eva_cluster_obj.cal_distortion_range(start=start, end=end, step=step)
        distortion_list = self.eva_cluster_obj.get_distortion_list()
        print(f'The distortion list is: {distortion_list}')
        self.eva_cluster_obj.draw_distortion_list()

    def cal_vmeasure(self, select_model, no_cluster):
        self.cal_latent_vec_for_model(select_model=select_model)
        self.label_correc_obj.set_decoder(decoder=self.model.get_encoder())
        self.label_correc_obj.set_maxpool_indices(maxpool_indices_array=self.train_maxpool_indices)
        self.clustering_obj.cal_cluster_centroid(all_latent_vec=self.all_latent_train_vec)
        cluster_centroids_idx = self.clustering_obj.get_cluster_centroid()
        print(f'The index of latent vector, who are cluster centroids: {cluster_centroids_idx}')
        self.label_correc_obj.dis_cluster_centroid(all_latent_vec=self.all_latent_train_vec,
                                                   cluster_centroid_idx=cluster_centroids_idx)
        label_map = self.label_correc_obj.get_label_map()
        self.transform_to_latent_vec_obj.cal_test_latent_vec(test_dataloader=self.load_data_obj.get_test_dataloader())
        all_latent_test_vec, test_labels = self.transform_to_latent_vec_obj.get_test_vec_data()
        true_labels = [label_map[int(label)] for label in test_labels]
        self.eva_cluster_obj.set_all_latent_vec(all_latent_vec=all_latent_test_vec)
        self.eva_cluster_obj.cal_vmeasure_score(no_cluster=no_cluster, true_labels=true_labels)

    def execute_input(self, mode, **kwargs):
        if mode == 'train_model':
            # train autoencoder
            self.execute_train_autoencoder(start_epoch=kwargs['start_epoch'],
                                           end_epoch=kwargs['end_epoch'])
            # train clustering model
            self.execute_train_cluster_model(select_model=kwargs['select_auto_model_clustering'])

        elif mode == 'eva_autoencoder':
            self.execute_eva_autoencoder(select_model=kwargs['select_auto_model_eva'])

        elif mode == 'cal_sil':
            self.cal_sil(select_model=kwargs['select_cluster_model_sil'],
                         start=kwargs['cluster_sil_start'],
                         end=kwargs['cluster_sil_end'],
                         step=kwargs['cluster_sil_step'])

        elif mode == 'cal_distortion':
            self.cal_dis(select_model=kwargs['select_cluster_model_dis'],
                         start=kwargs['cluster_dis_start'],
                         end=kwargs['cluster_dis_end'],
                         step=kwargs['cluster_dis_step'])

        elif mode == 'cal_vmeasure':
            self.cal_vmeasure(select_model=kwargs['select_cluster_model_vmeasure'],
                              no_cluster=kwargs['vmeasure_num_cluster'])


def user_input():
    pass


if __name__ == "__main__":

    pass

    # # cache_dir = './cache/'
    # # mem = Memory(location=cache_dir)
    #
    # autoencoder_model = Autoencoder(latent_dims=64)
    # autoencoder_adam_optimizer = optim.Adam(autoencoder_model.parameters(), lr=0.001)
    # mse_loss_func = nn.MSELoss()
    #
    # load_data_obj = LoadData(batch_size=128,
    #                          image_dim=(28, 28),
    #                          test_dataset_loc="D:/programming repo-1/autoencoder based unsupervised learning/data/test_dataset/",
    #                          train_dataset_loc="D:/programming repo-1/autoencoder based unsupervised learning/data/train_dataset/")
    # load_data_obj.load_dataset()
    # load_data_obj.create_dataloaders()
    #
    # train_autoencoder_obj = TrainAutoencoder(train_dataloader=load_data_obj.get_train_dataloader(),
    #                                          model=autoencoder_model,
    #                                          loss_func=mse_loss_func,
    #                                          optimizer=autoencoder_adam_optimizer,
    #                                          device='cuda')
    #
    # test_autoencoder_obj = ValidateAutoencoder(dataloader=load_data_obj.get_validation_dataloader(),
    #                                            loss_func=mse_loss_func,
    #                                            device='cuda')
    #
    # save_load_autoencoder_obj = SaveLoadAutoencoderModel(file_dir='autoencoder_models')
    #
    # plot_graph_obj = PlotAutoencoderGraph(device='cuda')
    #
    # clustering_obj = Clustering(no_cluster=10)
    #
    # # for epoch in range(1, 15):
    # #     # train model
    # #     train_autoencoder_obj.train()
    # #     model = train_autoencoder_obj.get_model()
    # #     train_loss = train_autoencoder_obj.get_train_loss()
    # #     # validate the model
    # #     test_autoencoder_obj.test(model=model)
    # #     test_loss = test_autoencoder_obj.get_test_loss()
    # #     # create a structure to store model information as well as the meta-data associated with it
    # #     model_name = f'autoencoder_model_{epoch}.pt'
    # #     encoder = model.get_encoder()
    # #     maxpool_indices = encoder.get_maxpool_indices()
    # #     model_data = {'model': model,
    # #                   'meta_data': {'train_loss': train_loss,
    # #                                 'test_loss': test_loss}}
    # #     # save the model information
    # #     save_load_autoencoder_obj.save(model=model_data, model_name=model_name)
    # #     # print the loss values for the model
    # #     print(f'''The Train loss is: {train_loss}\r
    # #     The test loss is: {test_loss}\r
    # #     Epoch: {epoch} is done.\n''')
    #
    # train_loss, test_loss, no_epochs = save_load_autoencoder_obj.get_model_data()
    #
    # # plot_graph_obj.plot_loss(no_epoch=no_epochs,
    # #                          loss_list=train_loss,
    # #                          title='Train Loss')
    # # plot_graph_obj.plot_loss(no_epoch=no_epochs,
    # #                          loss_list=test_loss,
    # #                          title='Test Loss')
    # plot_graph_obj.draw_image(dataloader=load_data_obj.get_validation_dataloader(),
    #                           model=save_load_autoencoder_obj.load(select_model=5)['model'])
    #
    # model_data = save_load_autoencoder_obj.load(select_model=5)
    # model = model_data['model']
    # encoder = model.get_encoder()
    # decoder = model.get_decoder()
    #
    # transform_to_latent_vec_obj = LatentVecConversion(device='cuda', model=model, latent_dim=64)
    # transform_to_latent_vec_obj.cal_latent_vec(dataloader=load_data_obj.get_train_dataloader())
    # all_latent_train_vec = transform_to_latent_vec_obj.get_all_latent_vec()  # use the cluster centers to find a point
    # train_maxpool_indices = transform_to_latent_vec_obj.get_maxpool_indices()
    #
    # transform_to_latent_vec_obj.cal_latent_vec(dataloader=load_data_obj.get_validation_dataloader())
    # all_latent_val_vec = transform_to_latent_vec_obj.get_all_latent_vec()
    #
    # transform_to_latent_vec_obj.cal_test_latent_vec(test_dataloader=load_data_obj.get_test_dataloader())
    # all_latent_test_vec, test_labels = transform_to_latent_vec_obj.get_test_vec_data()
    #
    # # disable later on: start
    # # temp_dict = {}
    # # for idx, label in enumerate(test_labels):
    # #     value = temp_dict.get(label, [])
    # #     value.append(idx)
    # #     temp_dict[label] = value
    # # plt.close()
    # # num_cluster = len(temp_dict)
    # # colour_list = cm.rainbow(np.linspace(0, 1, num_cluster))
    # # for point_idx_list, colour in zip(temp_dict.values(), colour_list):
    # #     plt.scatter(all_latent_train_vec[point_idx_list, 0], all_latent_train_vec[point_idx_list, 1], c=[colour, ])
    # # plt.show()
    # # disable later on: end
    #
    # clustering_obj.clustering_fit(all_latent_vec=all_latent_train_vec)
    # clustering_obj.clustering_predict(all_latent_vec=all_latent_train_vec)
    # pred_labels = clustering_obj.get_pred_labels()
    # label_point_idx_map = {}
    # for idx, label in enumerate(pred_labels):
    #     value = label_point_idx_map.get(label, [])
    #     value.append(idx)
    #     label_point_idx_map[label] = value
    # plot_graph_obj.draw_tsne(all_latent_vec=all_latent_train_vec,
    #                          label_point_idx_map=label_point_idx_map)
    #
    # clustering_obj.clustering_fit(all_latent_vec=all_latent_train_vec)
    # cluster_centers = clustering_obj.get_cluster_centers()
    # print(f'Cluster center shape: {cluster_centers.shape}')
    #
    # clustering_obj.cal_cluster_centroid(all_latent_vec=all_latent_train_vec)
    # cluster_centroids_idx = clustering_obj.get_cluster_centroid()
    # print(f'The index of latent vector, who are cluster centroids: {cluster_centroids_idx}')
    #
    # clustering_obj.clustering_predict(all_latent_vec=all_latent_val_vec)
    # labels = clustering_obj.get_pred_labels()
    # print(labels)
    #
    # # eva_cluster_obj = EvaluateClustering(Clustering=Clustering,
    # #                                      all_latent_vec=all_latent_train_vec,
    # #                                      true_labels=[])
    #
    # # eva_cluster_obj.cal_sil_score_range(start=2, end=25, step=1)
    # # sil_score_list = eva_cluster_obj.get_sil_score_list()
    # # print(f'The silhouette score list is: {sil_score_list}')
    # # eva_cluster_obj.draw_sil_score_list()
    #
    # # eva_cluster_obj.cal_distortion_range(start=2, end=50, step=1)
    # # distortion_list = eva_cluster_obj.get_distortion_list()
    # # print(f'The distortion list is: {distortion_list}')
    # # eva_cluster_obj.draw_distortion_list()
    #
    # label_correc_obj = LabelCorrection(decoder=decoder,
    #                                    maxpool_indices_array=train_maxpool_indices,
    #                                    device='cuda',
    #                                    latent_dim=64)
    # label_correc_obj.dis_cluster_centroid(all_latent_vec=all_latent_train_vec,
    #                                       cluster_centroid_idx=cluster_centroids_idx)
    # label_map = label_correc_obj.get_label_map()
    #
    # true_labels = [label_map[int(label)] for label in test_labels]
    #
    # eva_cluster_obj = EvaluateClustering(Clustering=Clustering,
    #                                      all_latent_vec=all_latent_test_vec,
    #                                      true_labels=true_labels)
    # eva_cluster_obj.cal_vmeasure_score(no_cluster=10)
    # vmeasure_score = eva_cluster_obj.get_vmeasure_score()
    # print(f'The v-measure score is: {vmeasure_score}')
