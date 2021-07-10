import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from models import unetnc, backwardmapper, full_model
from custom_dataset import CustomImageDataset_wc, Dataset_backward_mapping, Dataset_full_model
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import argparse
import sys
import json
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping


to_test = 'bm'
with open('config/test_config.json') as f:
    config = json.load(f)

DATA_PATH = "/home/sander/Inv3D_pre/inv3d/data/"

if to_test == 'wc':

    model = unetnc.Estimator3d(input_nc=3, output_nc=8, num_downs=5)
    model.load_state_dict(torch.load('models/pretrained/' + config['train_wc']['use_pretrained_model_name'] + '.pkl'))

    dataset_test = CustomImageDataset_wc(data_dir=DATA_PATH+'test/', transform=True, img_size=config['train_wc']['img_size'])
    test_loader = DataLoader(dataset_test, batch_size= config['train_wc']['batch_size_test'], num_workers=12)
    trainer = pl.Trainer(gpus=config['train_wc']['gpus'], max_epochs = config['train_wc']['max_epochs'],
                        log_every_n_steps=config['train_wc']['log_every_n_steps'])

    trainer.test(model, test_loader)    

elif to_test == 'bm':


    model_bm = backwardmapper.Backwardmapper()
    model_bm.load_state_dict(torch.load('models/pretrained/' + config['train_backwardmapper']['use_pretrained_model_name'] + '.pkl'))


    dataset_test = Dataset_backward_mapping(data_dir=DATA_PATH+'test/', img_size=config['train_backwardmapper']['img_size'])
    test_loader = DataLoader(dataset_test, batch_size= config['train_backwardmapper']['batch_size_test'], num_workers=12)

    trainer = pl.Trainer(gpus=config['train_backwardmapper']['gpus'], max_epochs = config['train_backwardmapper']['max_epochs'],
                        log_every_n_steps=config['train_backwardmapper']['log_every_n_steps'])

    trainer.test(model_bm, test_loader)

else:
    model = full_model.crease()
    model.load_state_dict(torch.load('models/pretrained/' + config['train_full']['use_pretrained_model_name'] + '.pkl'))

    dataset_test = Dataset_full_model(data_dir=DATA_PATH+'test/', img_size=config['train_full']['img_size'])
    test_loader = DataLoader(dataset_test, batch_size= config['train_full']['batch_size_test'], num_workers=12)

    trainer = pl.Trainer(gpus=config['train_full']['gpus'], max_epochs = config['train_full']['max_epochs'],
                log_every_n_steps=config['train_full']['log_every_n_steps'])

    trainer.test(model, test_loader)