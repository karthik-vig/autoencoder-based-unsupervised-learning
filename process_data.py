import os

import numpy as np
import torch
import torchvision.datasets as datasetLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode


class ImageLoader(Dataset):
    def __init__(self, root_path, transform):
        self.root_path = root_path
        self.image_list = []
        self.transform = transform
        dir_item_list = os.listdir('./data/')
        if 'processed_dataset.pt' in dir_item_list:
            self.image_list = torch.load(f='./data/processed_dataset.pt')
        else:
            dir_item_name_list = os.listdir(root_path)
            num_dir_item = len(dir_item_name_list)
            dir_item_name_array = np.array(dir_item_name_list)
            dir_idx_list = [i for i in range(num_dir_item)]
            np.random.shuffle(dir_idx_list)
            for dir_item_name in dir_item_name_array[dir_idx_list]:
                dir_item_path = os.path.join(root_path, dir_item_name)
                if os.path.isfile(dir_item_path) and ('jpg' == dir_item_name[-3:] or 'png' == dir_item_name[-3:]):
                    image = read_image(path=dir_item_path, mode=ImageReadMode.RGB)
                    transformed_image = self.transform(image)
                    transformed_image = transformed_image.double() / 255
                    self.image_list.append(transformed_image)
            torch.save(obj=self.image_list, f='./data/processed_dataset.pt')

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        return self.image_list[idx]


class LoadData:
    def __init__(self, batch_size, image_dim, test_dataset_loc, train_dataset_loc, seed=2022):
        self.batch_size = batch_size
        self.image_dim = image_dim
        self.seed = seed
        self.test_dataset_loc = test_dataset_loc
        self.train_dataset_loc = train_dataset_loc
        self.test_dataset = None
        self.train_dataset = None
        self.validation_dataset = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.validation_dataloader = None
        self.image_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize(size=self.image_dim)
                                                   ])

    def _split_dataset(self, dataset, split=0.85):
        # split dataset into training and validation:
        train_size = int(split * len(dataset))
        validation_size = len(dataset) - train_size

        train_dataset, validation_dataset = torch.utils.data.random_split(dataset,
                                                                          [train_size, validation_size],
                                                                          generator=torch.Generator().manual_seed(
                                                                              self.seed))
        return train_dataset, validation_dataset

    def load_dataset(self):
        # load the labelled datasets for training and validation:
        self.test_dataset = datasetLoader.ImageFolder(root=self.test_dataset_loc, transform=self.image_transform)
        dataset = ImageLoader(root_path=self.train_dataset_loc,
                              transform=transforms.Compose([
                                  transforms.Resize(size=self.image_dim)
                              ]))
        self.train_dataset, self.validation_dataset = self._split_dataset(dataset=dataset)
        labels_dict = {v: k for k, v in self.test_dataset.class_to_idx.items()}
        print(f"The test dataset labels are as following: {labels_dict}")

    def create_dataloaders(self):
        # dataloader for train and validation datasets:
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size)
        # sampler = sampler_func(train_dataset))
        self.validation_dataloader = torch.utils.data.DataLoader(self.validation_dataset,
                                                                 batch_size=self.batch_size)
        # dataloader for test dataset:
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset,
                                                           batch_size=self.batch_size)

    def get_train_dataset(self):
        return self.train_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_validation_dataset(self):
        return self.validation_dataset

    def get_train_dataloader(self):
        return self.train_dataloader

    def get_test_dataloader(self):
        return self.test_dataloader

    def get_validation_dataloader(self):
        return self.validation_dataloader


class TrainAutoencoder:
    def __init__(self, train_dataloader, model, loss_obj, optimizer, device):
        self.train_dataloader = train_dataloader
        self.model = model.to(device).double()
        self.loss_obj = loss_obj
        self.optimizer = optimizer
        self.device = device
        self.train_loss = None

    def train(self):
        self.train_loss = 0
        for data_batch in self.train_dataloader:
            data_batch = data_batch.to(self.device)
            output, loss = self.model(data_batch)
            # kl_div_val = self.model.get_kl_div_val()
            # recon_loss = self.loss_obj.mse_loss(x_hat=output, x=data_batch)
            # mean_x, std_x = self.model.get_encoder().get_mean_std()
            # latent_vec = self.model.get_latent_vec()
            # kl_div_val = self.loss_obj.kl_div(x=latent_vec, mean=mean_x, sigma=std_x)
            # loss = (kl_div_val - recon_loss).sum()
            # loss = self.model.get_loss()
            self.train_loss += loss.clone().detach()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def get_model(self):
        return self.model

    def get_train_loss(self):
        return self.train_loss


class SaveLoadAutoencoderModel:
    def __init__(self, file_dir):
        self.file_dir = file_dir
        self._create_dir()

    def _create_dir(self):
        dir_list = os.listdir('./')
        if self.file_dir not in dir_list:
            os.mkdir(self.file_dir)

    def save(self, model, model_name):
        model_path = os.path.join(self.file_dir, model_name)
        torch.save(obj=model, f=model_path)

    def load(self, select_model=None):
        model_name_list = os.listdir(os.path.join('.', self.file_dir))
        model_name_list.sort()
        if select_model == None:
            for idx, model_name in enumerate(model_name_list):
                print(f'{idx}) {model_name}\n')
            select_model = int(input('Enter a corresponding number: '))
            if select_model < 0 or select_model >= len(model_name_list):
                return None
        model_path = os.path.join(self.file_dir, model_name_list[select_model])
        model = torch.load(f=model_path)
        return model

    def load_all(self):
        model_name_list = os.listdir(os.path.join('.', self.file_dir))
        model_name_list.sort()
        model_list = []
        for model_name in model_name_list:
            model_path = os.path.join(self.file_dir, model_name)
            model = torch.load(f=model_path)
            model_list.append(model)
        return model_list

    def get_model_data(self):
        model_data_list = self.load_all()
        train_loss = []
        test_loss = []
        for model_data in model_data_list:
            train_loss.append(model_data['meta_data']['train_loss'].cpu())
            test_loss.append(model_data['meta_data']['test_loss'].cpu())
        return train_loss, test_loss, len(model_data_list)


class LatentVecConversion:
    def __init__(self, device, latent_dim, maxpool_size):
        self.device = device
        self.model = None
        self.all_latent_vec = None
        self.all_test_latent_vec = None
        self.maxpool_indices_array = None
        self.test_maxpool_indices_array = None
        self.test_labels = []
        self.latent_dim = latent_dim
        self.maxpool_size = maxpool_size

    def set_model(self, model):
        self.model = model.to(self.device).double()

    def cal_latent_vec(self, dataloader):
        print(f'Calculating Latent vector')
        all_latent_vec = torch.zeros((1, self.latent_dim), dtype=torch.float64).to(self.device)
        self.maxpool_indices_array = torch.zeros(self.maxpool_size, dtype=torch.int64).to(self.device)
        encoder = self.model.get_encoder()
        for data_batch in dataloader:
            data_batch = data_batch.to(self.device)
            output = encoder(data_batch)
            maxpool_indices = encoder.get_maxpool_indices()
            latent_vec = torch.flatten(output, 1).detach()
            all_latent_vec = torch.vstack((all_latent_vec, latent_vec))
            self.maxpool_indices_array = torch.vstack((self.maxpool_indices_array, maxpool_indices))
        self.all_latent_vec = all_latent_vec[1:, :].cpu().numpy()
        self.maxpool_indices_array = self.maxpool_indices_array[1:].cpu()

    def cal_test_latent_vec(self, test_dataloader):
        print(f'Calculating test latent vec')
        self.test_labels = []
        all_test_latent_vec = torch.zeros((1, self.latent_dim), dtype=torch.float64).to(self.device)
        self.test_maxpool_indices_array = torch.zeros(self.maxpool_size, dtype=torch.int64).to(self.device)
        encoder = self.model.get_encoder()
        for data_batch, label in test_dataloader:
            self.test_labels += label.tolist()
            data_batch = data_batch.double()
            data_batch = data_batch.to(self.device)
            output = encoder(data_batch)
            maxpool_indices = encoder.get_maxpool_indices()
            latent_vec = torch.flatten(output, 1).detach()
            all_test_latent_vec = torch.vstack((all_test_latent_vec, latent_vec))
            self.test_maxpool_indices_array = torch.vstack((self.maxpool_indices_array, maxpool_indices.cpu()))
        self.all_test_latent_vec = all_test_latent_vec[1:, :].cpu().numpy()
        self.test_maxpool_indices_array = self.maxpool_indices_array[1:].cpu()

    def get_all_latent_vec(self):
        return self.all_latent_vec

    def get_maxpool_indices(self):
        return self.maxpool_indices_array

    def get_test_maxpool_indices(self):
        return self.test_maxpool_indices_array

    def get_test_vec_data(self):
        return self.all_test_latent_vec, self.test_labels
