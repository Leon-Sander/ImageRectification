import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from models import unetnc, backwardmapper, full_model
from custom_dataset import CustomImageDataset_wc, Dataset_backward_mapping, Dataset_full_model
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import argparse
import json
import torch
import sys

def main():

    with open('config/test_config.json') as f:
        config = json.load(f)
    if args['model'] == 'train_wc':


        DATA_PATH = config['train_wc']['data_path']

        model = unetnc.Estimator3d(num_downs = config['train_full']['num_downs'], 
                                input_nc_wc = config['train_full']['input_nc_wc'], 
                                output_nc_wc = config['train_full']['output_nc_wc'], 
                                img_size = config['train_full']['img_size'],
                                angle_loss_type=config['train_wc']['angle_loss_type'])

        model.load_state_dict(torch.load('models/pretrained/' + config['train_wc']['use_pretrained_model_name'] + '.pkl'))

        dataset_test = CustomImageDataset_wc(data_dir=DATA_PATH+'test/', transform=True, img_size=config['train_wc']['img_size'])
        test_loader = DataLoader(dataset_test, batch_size= config['train_wc']['batch_size_test'], num_workers=12)
        trainer = pl.Trainer(gpus=config['train_wc']['gpus'], max_epochs = config['train_wc']['max_epochs'],
                            log_every_n_steps=config['train_wc']['log_every_n_steps'])

        trainer.test(model, test_loader)    

    elif args['model'] == 'train_backwardmapper':
        DATA_PATH = config['train_backwardmapper']['data_path']
        
        model_bm = backwardmapper.Backwardmapper(angle_loss_type=config['train_backwardmapper']['angle_loss_type'])
        model_bm.load_state_dict(torch.load('models/pretrained/' + config['train_backwardmapper']['use_pretrained_model_name'] + '.pkl'))


        dataset_test = Dataset_backward_mapping(data_dir=DATA_PATH+'test/', img_size=config['train_backwardmapper']['img_size'],
                                                    resizing_from_size=config['train_backwardmapper']['resizing_from_size'])
        test_loader = DataLoader(dataset_test, batch_size= config['train_backwardmapper']['batch_size_test'], num_workers=12)

        trainer = pl.Trainer(gpus=config['train_backwardmapper']['gpus'], max_epochs = config['train_backwardmapper']['max_epochs'],
                            log_every_n_steps=config['train_backwardmapper']['log_every_n_steps'])

        trainer.test(model_bm, test_loader)

    elif args['model'] == 'train_full':
        DATA_PATH = config['train_full']['data_path']

        model = full_model.crease(angle_loss_type=config['train_full']['angle_loss_type'])
        model.load_state_dict(torch.load('models/pretrained/' + config['train_full']['use_pretrained_model_name'] + '.pkl'))

        dataset_test = Dataset_full_model(data_dir=DATA_PATH+'test/', img_size=config['train_full']['img_size'],
                                            resizing_from_size=config['train_full']['resizing_from_size'])
        test_loader = DataLoader(dataset_test, batch_size= config['train_full']['batch_size_test'], num_workers=12)

        trainer = pl.Trainer(gpus=config['train_full']['gpus'], max_epochs = config['train_full']['max_epochs'],
                    log_every_n_steps=config['train_full']['log_every_n_steps'])

        trainer.test(model, test_loader)
    
    else:
        print('You have to specify one of the models train_wc, train_backwardmapper, train_full')
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test')   
    parser.add_argument('-m','--model', help='models are train_wc, train_backwardmapper, train_full', required=True)
    
    args = vars(parser.parse_args())
    main(args)