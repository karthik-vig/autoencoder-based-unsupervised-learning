import json
import os

import torch.nn as nn
import torch.optim as optim

from evaluation import ValidateAutoencoder, PlotAutoencoderGraph, EvaluateClustering, LabelCorrection
from ml_models import Autoencoder, Clustering
from process_data import LoadData, TrainAutoencoder, SaveLoadAutoencoderModel, LatentVecConversion


class Execute:
    def __init__(self, kwargs):
        self.autoencoder_model = Autoencoder(latent_dims=kwargs['latent_dim'], autoencoder_setup=kwargs['autoencoder'])
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

        self.transform_to_latent_vec_obj = LatentVecConversion(device=kwargs['device'], latent_dim=kwargs['latent_dim'],
                                                               maxpool_size=kwargs['maxpool_size'])

        self.eva_cluster_obj = EvaluateClustering(Clustering=Clustering)

        self.label_correc_obj = LabelCorrection(device=kwargs['device'],
                                                latent_dim=kwargs['latent_dim'],
                                                maxpool_size=kwargs['maxpool_size'])

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
        num_digit = len(str(end_epoch))
        end_epoch += 1
        for epoch in range(start_epoch, end_epoch):
            # train model
            self.train_autoencoder_obj.train()
            model = self.train_autoencoder_obj.get_model()
            train_loss = self.train_autoencoder_obj.get_train_loss()
            # validate the model
            self.test_autoencoder_obj.test(model=model)
            test_loss = self.test_autoencoder_obj.get_test_loss()
            # create a structure to store model information as well as the meta-data associated with it
            curr_num_digit = len(str(epoch))
            zero_pad = '0' * (num_digit - curr_num_digit)
            model_name = f'autoencoder_model_{zero_pad}{epoch}.pt'
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
        label_point_idx_map = {}
        for idx, label in enumerate(self.train_pred_labels):
            value = label_point_idx_map.get(label, [])
            value.append(idx)
            label_point_idx_map[label] = value
        self.plot_graph_obj.draw_tsne(all_latent_vec=self.all_latent_train_vec,
                                      label_point_idx_map=label_point_idx_map)

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
        model = self.model_data['model']
        self.label_correc_obj.set_decoder(decoder=model.get_decoder())
        self.label_correc_obj.set_maxpool_indices(maxpool_indices_array=self.train_maxpool_indices)
        self.cal_latent_vec_for_model(select_model=select_model)
        self.clustering_obj.clustering_fit(all_latent_vec=self.all_latent_train_vec)
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
        v_score = self.eva_cluster_obj.get_vmeasure_score()
        print(f'The v-measure score is: {v_score}')

    def execute_input(self, mode, kwargs):
        if mode == 'train_autoencoder':
            self.execute_train_autoencoder(start_epoch=1,
                                           end_epoch=kwargs['end_epoch'])

        elif mode == 'train_cluster':
            self.execute_train_cluster_model(select_model=kwargs['select_auto_model_clustering'])
            print(f'The predicted training dataset labels are : {self.train_pred_labels}')

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


def get_setup_json():
    setup_json = {'autoencoder': {'encoder': {'conv1': {'in_ch': 3,
                                                        'out_ch': 10,
                                                        'kernel_sz': 3,
                                                        'stride': 1},
                                              'conv2': {'in_ch': 10,
                                                        'out_ch': 20,
                                                        'kernel_sz': 3,
                                                        'stride': 1},
                                              'conv3': {'in_ch': 20,
                                                        'out_ch': 30,
                                                        'kernel_sz': 3,
                                                        'stride': 1},
                                              'pool': {'kernel_sz': 2,
                                                       'stride': 2},
                                              'ln1': {'in_feat': 30 * 11 * 11,
                                                      'out_feat': 1024},
                                              'ln2': {'in_feat': 1024,
                                                      'out_feat': 512},
                                              'ln3': {'in_feat': 512,
                                                      'out_feat': 256},
                                              'ln4': {'in_feat': 256}},
                                  'decoder': {'ln1': {'out_feat': 256},
                                              'ln2': {'in_feat': 256,
                                                      'out_feat': 512},
                                              'ln3': {'in_feat': 512,
                                                      'out_feat': 1024},
                                              'ln4': {'in_feat': 1024,
                                                      'out_feat': 30 * 11 * 11},
                                              'unflatten': {'sz': [30, 11, 11]},
                                              'invpool': {'kernel_sz': 2,
                                                          'stride': 2},
                                              'deconv1': {'in_ch': 30,
                                                          'out_ch': 20,
                                                          'kernel_sz': 3,
                                                          'stride': 1},
                                              'deconv2': {'in_ch': 20,
                                                          'out_ch': 10,
                                                          'kernel_sz': 3,
                                                          'stride': 1},
                                              'deconv3': {'in_ch': 10,
                                                          'out_ch': 3,
                                                          'kernel_sz': 3,
                                                          'stride': 1}}},
                  'maxpool_size': [1, 30, 11, 11],
                  'latent_dim': 100,
                  'lr': 0.001,
                  'batch_size': 128,
                  'img_dim': [28, 28],
                  'device': 'cuda',
                  'num_cluster': 10
                  }
    if 'setup.json' in os.listdir('./'):
        with open('setup.json', 'r') as json_file:
            setup_json = json.load(json_file)
    else:
        with open('setup.json', 'w') as json_file:
            json.dump(setup_json, json_file)
    return setup_json


def select_autoencoder_model():
    file_name_list = os.listdir('./autoencoder_models/')
    file_name_list.sort()
    for idx, file_name in enumerate(file_name_list, 1):
        print(f'{idx}) {file_name}')
    input_val = int(input('Which autoencoder model to user for train cluster model? : '))
    if input_val <= 0 or input_val > len(file_name_list):
        return None
    return input_val - 1


def user_train_auto_input():
    user_train_auto_input_map = {'end_epoch': int(input('Enter the number of epochs: '))}
    return user_train_auto_input_map


def user_train_cluser_input():
    user_train_cluser_input_map = {'select_auto_model_clustering': select_autoencoder_model()}
    return user_train_cluser_input_map


def user_eva_autoencoder_input():
    user_eva_autoencoder_input_map = {'select_auto_model_eva': select_autoencoder_model()}
    return user_eva_autoencoder_input_map


def user_cal_sil_input():
    user_cal_sil_input_map = {'select_cluster_model_sil': select_autoencoder_model(),
                              'cluster_sil_start': int(input('Enter the start number of cluster: ')),
                              'cluster_sil_end': int(input('Enter the end number of cluster: ')),
                              'cluster_sil_step': int(input('Enter the step to increase number of clusters: '))}
    return user_cal_sil_input_map


def user_cal_dis_input():
    user_cal_dis_input_map = {'select_cluster_model_dis': select_autoencoder_model(),
                              'cluster_dis_start': int(input('Enter the start number of cluster: ')),
                              'cluster_dis_end': int(input('Enter the end number of cluster: ')),
                              'cluster_dis_step': int(input('Enter the step to increase number of clusters: '))}
    return user_cal_dis_input_map


def user_cal_vmeasure_input():
    user_cal_vmeasure_input_map = {'select_cluster_model_vmeasure': select_autoencoder_model(),
                                   'vmeasure_num_cluster': int(input('Enter the number of clusters: '))}
    return user_cal_vmeasure_input_map


def user_input():
    user_input_value = {}
    setup_info = get_setup_json()
    mode_map = {1: 'train_autoencoder',
                2: 'train_cluster',
                3: 'eva_autoencoder',
                4: 'cal_sil',
                5: 'cal_distortion',
                6: 'cal_vmeasure'}
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    curr_dir = curr_dir.replace("""\\""", """/""")
    setup_info['train_dataset_loc'] = curr_dir + '/data/train_dataset'
    setup_info['test_dataset_loc'] = curr_dir + '/data/test_dataset'
    # print(setup_info)
    print('Enter Mode: ')
    for key, value in mode_map.items():
        print(f'{key}) {value}')
    mode_input = int(input('Choose any one option or enter random number to exit: '))
    if mode_input <= 0 or mode_input > len(mode_map):
        return None
    mode_input = mode_map[mode_input]
    exe_info_input = None
    if mode_input == 'train_autoencoder':
        exe_info_input = user_train_auto_input()
    elif mode_input == 'train_cluster':
        exe_info_input = user_train_cluser_input()
    elif mode_input == 'eva_autoencoder':
        exe_info_input = user_eva_autoencoder_input()
    elif mode_input == 'cal_sil':
        exe_info_input = user_cal_sil_input()
    elif mode_input == 'cal_distortion':
        exe_info_input = user_cal_dis_input()
    elif mode_input == 'cal_vmeasure':
        exe_info_input = user_cal_vmeasure_input()
    user_input_value['execution_info'] = exe_info_input
    user_input_value['setup_info'] = setup_info
    user_input_value['mode'] = mode_input
    return user_input_value


def main():
    user_input_value = user_input()
    if not user_input_value:
        return None
    exe_obj = Execute(kwargs=user_input_value['setup_info'])
    exe_obj.execute_input(mode=user_input_value['mode'],
                          kwargs=user_input_value['execution_info'])


if __name__ == "__main__":
    main()
