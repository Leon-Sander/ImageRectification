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


def main(args):

    
    with open('config/train_config.json') as f:
        config = json.load(f)

    if args['model'] == 'train_wc':
        DATA_PATH = config['train_wc']['data_path']
        
        model = unetnc.Estimator3d(input_nc=config['train_wc']['input_nc'],
                                    output_nc=config['train_wc']['output_nc'], 
                                    num_downs=config['train_wc']['num_downs'], 
                                    ngf = config['train_wc']['ngf'],
                                    use_dropout= bool(config['train_wc']['use_dropout']),
                                    lr = config['train_wc']['lr'],
                                    weight_decay = config['train_wc']['weight_decay'],
                                    angle_loss_type=config['train_wc']['angle_loss_type'])
        if bool(config['train_wc']['use_pretrained']):
            model.load_state_dict(torch.load('models/pretrained/' + config['train_wc']['use_pretrained_model_name'] + '.pkl'))

        dataset_train = CustomImageDataset_wc(data_dir=DATA_PATH+'train/', transform=True, img_size=config['train_wc']['img_size'])
        dataset_val = CustomImageDataset_wc(data_dir=DATA_PATH+'val/', transform=True, img_size=config['train_wc']['img_size'])
        dataset_test = CustomImageDataset_wc(data_dir=DATA_PATH+'test/', transform=True, img_size=config['train_wc']['img_size'])

        train_loader = DataLoader(dataset_train, batch_size= config['train_wc']['batch_size_train'], num_workers=12)
        val_loader = DataLoader(dataset_val, batch_size= config['train_wc']['batch_size_val'], num_workers=12)
        test_loader = DataLoader(dataset_test, batch_size= config['train_wc']['batch_size_test'], num_workers=12)

        early_stop_callback = EarlyStopping(monitor='validation_loss', min_delta=0.001, patience=config['train_wc']['early_stopping_patience'], verbose=True, mode='min')
        trainer = pl.Trainer(gpus=config['train_wc']['gpus'], max_epochs = config['train_wc']['max_epochs'],
                            log_every_n_steps=config['train_wc']['log_every_n_steps'],
                            check_val_every_n_epoch = config['train_wc']['check_val_every_n_epoch'],
                            callbacks=early_stop_callback)
        trainer.fit(model, train_loader, val_loader)
        torch.save(model.state_dict(), 'models/pretrained/' + config['train_wc']['save_name'] + '.pkl')
        trainer.test(model, test_loader)

    elif args['model'] == 'train_backwardmapper':
        DATA_PATH = config['train_backwardmapper']['data_path']

        model_bm = backwardmapper.Backwardmapper(img_size=config['train_backwardmapper']['img_size'], 
                                                in_channels=config['train_backwardmapper']['in_channels'], 
                                                out_channels=config['train_backwardmapper']['out_channels'], 
                                                filters=config['train_backwardmapper']['filters'],
                                                fc_units=config['train_backwardmapper']['fc_units'], 
                                                lr = config['train_backwardmapper']['lr'], 
                                                weight_decay=config['train_backwardmapper']['weight_decay'],
                                                angle_loss_type=config['train_backwardmapper']['angle_loss_type'])

        if bool(config['train_backwardmapper']['use_pretrained']):

            model_bm.load_state_dict(torch.load('models/pretrained/' + config['train_backwardmapper']['use_pretrained_model_name'] + '.pkl'))
        train_dataset_bm = Dataset_backward_mapping(data_dir=DATA_PATH+'train/', img_size=config['train_backwardmapper']['img_size'],
                                                    resizing_from_size=config['train_backwardmapper']['resizing_from_size'])
        dataset_val = Dataset_backward_mapping(data_dir=DATA_PATH+'val/', img_size=config['train_backwardmapper']['img_size'],
                                                resizing_from_size=config['train_backwardmapper']['resizing_from_size'])
        dataset_test = Dataset_backward_mapping(data_dir=DATA_PATH+'test/', img_size=config['train_backwardmapper']['img_size'],
                                                resizing_from_size=config['train_backwardmapper']['resizing_from_size'])
        
        
        val_loader = DataLoader(dataset_val, batch_size= config['train_backwardmapper']['batch_size_val'], num_workers=12)
        test_loader = DataLoader(dataset_test, batch_size= config['train_backwardmapper']['batch_size_test'], num_workers=12)
        train_loader = DataLoader(train_dataset_bm, batch_size= config['train_backwardmapper']['batch_size_train'], num_workers=12)
        
        early_stop_callback = EarlyStopping(monitor='validation_loss', min_delta=0.001, patience=config['train_backwardmapper']['early_stopping_patience'],
                                             verbose=True, mode='min')
        #logger = TensorBoardLogger("tb_logs", name=config['train_backwardmapper']['save_name'])
        trainer = pl.Trainer(gpus=config['train_backwardmapper']['gpus'], max_epochs = config['train_backwardmapper']['max_epochs'],
                            log_every_n_steps=config['train_backwardmapper']['log_every_n_steps'],
                            check_val_every_n_epoch = config['train_backwardmapper']['check_val_every_n_epoch'],
                            callbacks=early_stop_callback)
                            #resume_from_checkpoint = '/home/sander/code/ImageRectification/lightning_logs/version_10/checkpoints/epoch=71-step=7559.ckpt')

        trainer.fit(model_bm, train_loader, val_loader)
        torch.save(model_bm.state_dict(), 'models/pretrained/' + config['train_backwardmapper']['save_name'] + '.pkl')
        trainer.test(model_bm, test_loader)

    elif args['model'] == 'train_full':
        DATA_PATH = config['train_full']['data_path']

        model = full_model.crease(num_downs = config['train_full']['num_downs'], 
                                input_nc_wc = config['train_full']['input_nc_wc'], 
                                output_nc_wc = config['train_full']['output_nc_wc'], 
                                img_size = config['train_full']['img_size'], 
                                use_pre_trained = bool(config['train_full']['use_pre_trained']), 
                                ngf_wc=config['train_full']['ngf_wc'],
                                use_dropout=bool(config['train_full']['use_dropout']), 
                                lr = config['train_full']['lr'], 
                                weight_decay=config['train_full']['weight_decay'], 
                                load_3d =config['train_full']['load_3d'],
                                load_bm = config['train_full']['load_bm'],
                                angle_loss_type=config['train_full']['angle_loss_type'])

        if bool(config['train_full']['use_pretrained_crease']):
            model.load_state_dict(torch.load('models/pretrained/' + config['train_full']['use_pretrained_model_name'] + '.pkl'))

        early_stop_callback = EarlyStopping(monitor='validation_loss', min_delta=0.001, patience=config['train_full']['early_stopping_patience'],
                                        verbose=True, mode='min')


        dataset_train = Dataset_full_model(data_dir=DATA_PATH+'train/', img_size=config['train_full']['img_size'],
                                            resizing_from_size=config['train_full']['resizing_from_size'])
        dataset_val = Dataset_full_model(data_dir=DATA_PATH+'val/', img_size=config['train_full']['img_size'],
                                            resizing_from_size=config['train_full']['resizing_from_size'])
        dataset_test = Dataset_full_model(data_dir=DATA_PATH+'test/', img_size=config['train_full']['img_size'],
                                            resizing_from_size=config['train_full']['resizing_from_size'])
        

        train_loader = DataLoader(dataset_train, batch_size= config['train_full']['batch_size_train'], num_workers=12)
        val_loader = DataLoader(dataset_val, batch_size= config['train_full']['batch_size_val'], num_workers=12)
        test_loader = DataLoader(dataset_test, batch_size= config['train_full']['batch_size_test'], num_workers=12)

        trainer = pl.Trainer(gpus=config['train_full']['gpus'], max_epochs = config['train_full']['max_epochs'],
                    log_every_n_steps=config['train_full']['log_every_n_steps'],
                    check_val_every_n_epoch = config['train_full']['check_val_every_n_epoch'],
                    callbacks=early_stop_callback)

        trainer.fit(model, train_loader, val_loader)
        torch.save(model.state_dict(), 'models/pretrained/' + config['train_full']['save_name'] + '.pkl')
        trainer.test(model, test_loader)

    else:
        print('You have to specify one of the models train_wc, train_backwardmapper, train_full')
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')   
    parser.add_argument('-m','--model', help='models are train_wc, train_backwardmapper, train_full', required=True)
    
    args = vars(parser.parse_args())
    main(args)