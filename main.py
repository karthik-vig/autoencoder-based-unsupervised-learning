import torch.optim as optim
import torch.nn as nn
from ml_models import Autoencoder, Clustering
from process_data import LoadData, TrainAutoencoder, SaveLoadAutoencoderModel, LatentVecConversion
from evaluation import ValidateAutoencoder, PlotAutoencoderGraph, EvaluateClustering, LabelCorrection

def execute(**kwargs):
    pass

def user_input():
    pass

if __name__ == "__main__":
    autoencoder_model = Autoencoder(latent_dims=64)
    autoencoder_adam_optimizer = optim.Adam(autoencoder_model.parameters(), lr=0.001)
    mse_loss_func = nn.MSELoss()

    load_data_obj = LoadData(batch_size=128,
                             image_dim=(28, 28),
                             test_dataset_loc="D:/programming repo-1/autoencoder based unsupervised learning/data/test_dataset/",
                             train_dataset_loc="D:/programming repo-1/autoencoder based unsupervised learning/data/train_dataset/")
    load_data_obj.load_dataset()
    load_data_obj.create_dataloaders()

    train_autoencoder_obj = TrainAutoencoder(train_dataloader=load_data_obj.get_train_dataloader(),
                                             model=autoencoder_model,
                                             loss_func=mse_loss_func,
                                             optimizer=autoencoder_adam_optimizer,
                                             device='cuda')

    test_autoencoder_obj = ValidateAutoencoder(dataloader=load_data_obj.get_validation_dataloader(),
                                               loss_func=mse_loss_func,
                                               device='cuda')

    save_load_autoencoder_obj = SaveLoadAutoencoderModel(file_dir='autoencoder_models')

    plot_graph_obj = PlotAutoencoderGraph(device='cuda')

    clustering_obj = Clustering(no_cluster=10)

    # for epoch in range(1, 50):
    #     # train model
    #     train_autoencoder_obj.train()
    #     model = train_autoencoder_obj.get_model()
    #     train_loss = train_autoencoder_obj.get_train_loss()
    #     # validate the model
    #     test_autoencoder_obj.test(model=model)
    #     test_loss = test_autoencoder_obj.get_test_loss()
    #     # create a structure to store model information as well as the meta-data associated with it
    #     model_name = f'autoencoder_model_{epoch}.pt'
    #     encoder = model.get_encoder()
    #     maxpool_indices = encoder.get_maxpool_indices()
    #     model_data = {'model': model,
    #                   'meta_data': {'train_loss': train_loss,
    #                                 'test_loss': test_loss}}
    #     # save the model information
    #     save_load_autoencoder_obj.save(model=model_data, model_name=model_name)
    #     # print the loss values for the model
    #     print(f'''The Train loss is: {train_loss}\r
    #     The test loss is: {test_loss}\r
    #     Epoch: {epoch} is done.\n''')

    # train_loss, test_loss, no_epochs = save_load_autoencoder_obj.get_model_data()

    # plot_graph_obj.plot_loss(no_epoch=no_epochs,
    #                          loss_list=train_loss,
    #                          title='Train Loss')
    # plot_graph_obj.plot_loss(no_epoch=no_epochs,
    #                          loss_list=test_loss,
    #                          title='Test Loss')
    # plot_graph_obj.draw_image(dataloader=load_data_obj.get_validation_dataloader(),
    #                           model=save_load_autoencoder_obj.load(select_model=42)['model'])

    model_data = save_load_autoencoder_obj.load(select_model=42)
    model = model_data['model']
    encoder = model.get_encoder()
    decoder = model.get_decoder()

    transform_to_latent_vec_obj = LatentVecConversion(device='cuda', model=model)
    transform_to_latent_vec_obj.cal_latent_vec(dataloader=load_data_obj.get_train_dataloader())
    all_latent_train_vec = transform_to_latent_vec_obj.get_all_latent_vec() # use the cluster centers to find a point
    train_maxpool_indices = transform_to_latent_vec_obj.get_maxpool_indices()
    # close to the cluster. using this point (by either manual labelling or using its given label from the dataset)
    # map the label given by the cluster to the original label. using this now perform correct prediction as well as
    # measure v-measure score.

    transform_to_latent_vec_obj.cal_latent_vec(dataloader=load_data_obj.get_validation_dataloader())
    all_latent_val_vec = transform_to_latent_vec_obj.get_all_latent_vec()

    transform_to_latent_vec_obj.cal_test_latent_vec(test_dataloader=load_data_obj.get_test_dataloader())
    all_latent_test_vec, test_labels = transform_to_latent_vec_obj.get_test_vec_data()

    clustering_obj.clustering_fit(all_latent_vec=all_latent_train_vec)
    clustering_obj.clustering_predict(all_latent_vec=all_latent_train_vec)
    pred_labels = clustering_obj.get_pred_labels()
    label_point_idx_map = {}
    for idx, label in enumerate(pred_labels):
        value = label_point_idx_map.get(label, [])
        value.append(idx)
        label_point_idx_map[label] = value
    plot_graph_obj.draw_tsne(all_latent_vec=all_latent_train_vec,
                             label_point_idx_map=label_point_idx_map)

    clustering_obj.clustering_fit(all_latent_vec=all_latent_train_vec)
    cluster_centers = clustering_obj.get_cluster_centers()
    print(f'Cluster center shape: {cluster_centers.shape}')

    clustering_obj.cal_cluster_centroid(all_latent_vec=all_latent_train_vec)
    cluster_centroids_idx = clustering_obj.get_cluster_centroid()
    print(f'The index of latent vector, who are cluster centroids: {cluster_centroids_idx}')

    clustering_obj.clustering_predict(all_latent_vec=all_latent_val_vec)
    labels = clustering_obj.get_pred_labels()
    print(labels)

    # eva_cluster_obj = EvaluateClustering(Clustering=Clustering,
    #                                      all_latent_vec=all_latent_train_vec,
    #                                      true_labels=[])

    # eva_cluster_obj.cal_sil_score_range(start=2, end=25, step=1)
    # sil_score_list = eva_cluster_obj.get_sil_score_list()
    # print(f'The silhouette score list is: {sil_score_list}')
    # eva_cluster_obj.draw_sil_score_list()

    # eva_cluster_obj.cal_distortion_range(start=2, end=50, step=1)
    # distortion_list = eva_cluster_obj.get_distortion_list()
    # print(f'The distortion list is: {distortion_list}')
    # eva_cluster_obj.draw_distortion_list()

    label_correc_obj = LabelCorrection(decoder=decoder, maxpool_indices_array=train_maxpool_indices, device='cuda')
    label_correc_obj.dis_cluster_centroid(all_latent_vec=all_latent_train_vec,
                                          cluster_centroid_idx=cluster_centroids_idx)
    label_map = label_correc_obj.get_label_map()

    true_labels = [label_map[int(label)] for label in test_labels]

    eva_cluster_obj = EvaluateClustering(Clustering=Clustering,
                                         all_latent_vec=all_latent_test_vec,
                                         true_labels=true_labels)
    eva_cluster_obj.cal_vmeasure_score(no_cluster=10)
    vmeasure_score = eva_cluster_obj.get_vmeasure_score()
    print(f'The v-measure score is: {vmeasure_score}')






