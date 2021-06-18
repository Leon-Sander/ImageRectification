from models import unetnc, backwardmapper, full_model
from custom_dataset import CustomImageDataset_wc, Dataset_backward_mapping, Dataset_full_model
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import argparse
import sys
import json
import torch

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
                                    weight_decay = config['train_wc']['weight_decay'])
        #model.load_state_dict(torch.load('models/pretrained/wc_test2.pkl'))

        dataset_train = CustomImageDataset_wc(data_dir=DATA_PATH+'train/', transform=True)
        #dataset_val = CustomImageDataset_wc(data_dir=DATA_PATH+'val/', transform=True)
        #dataset_test = CustomImageDataset_wc(data_dir=DATA_PATH+'test/', transform=True)

        train_loader = DataLoader(dataset_train, batch_size= config['train_wc']['batch_size_train'], num_workers=12)
        #val_loader = DataLoader(dataset_val, batch_size= config['train_wc']['batch_size_train'], num_workers=12)
        #test_loader = DataLoader(dataset_test, batch_size= config['train_wc']['batch_size_train'], num_workers=12)


        trainer = pl.Trainer(gpus=config['train_wc']['gpus'], max_epochs = config['train_wc']['max_epochs'])
        trainer.fit(model, train_loader)
        torch.save(model.state_dict(), 'models/pretrained/' + config['train_wc']['save_name'] + '.pkl')


    elif args['model'] == 'train_backwardmapper':
        DATA_PATH = config['train_backwardmapper']['data_path']

        model_bm = backwardmapper.Backwardmapper(img_size=config['train_backwardmapper']['img_size'], 
                                                in_channels=config['train_backwardmapper']['in_channels'], 
                                                out_channels=config['train_backwardmapper']['out_channels'], 
                                                filters=config['train_backwardmapper']['filters'],
                                                fc_units=config['train_backwardmapper']['fc_units'], 
                                                lr = config['train_backwardmapper']['lr'], 
                                                weight_decay=config['train_backwardmapper']['weight_decay'])

        if bool(config['train_backwardmapper']['use_pre_trained']):

            model_bm.load_state_dict(torch.load('models/pretrained/' + config['train_backwardmapper']['use_pretrained_model_name'] + '.pkl'))
        train_dataset_bm = Dataset_backward_mapping(data_dir=DATA_PATH+'train/')
        #dataset_val = CustomImageDataset_wc(data_dir=DATA_PATH+'val/', transform=True)
        #dataset_test = CustomImageDataset_wc(data_dir=DATA_PATH+'test/', transform=True)
        
        
        #val_loader = DataLoader(dataset_val, batch_size= config['train_wc']['batch_size_train'], num_workers=12)
        #test_loader = DataLoader(dataset_test, batch_size= config['train_wc']['batch_size_train'], num_workers=12)
        train_loader = DataLoader(train_dataset_bm, batch_size= config['train_backwardmapper']['batch_size_train'], num_workers=12)
        
        trainer = pl.Trainer(gpus=config['train_backwardmapper']['gpus'], max_epochs = config['train_backwardmapper']['max_epochs'])
        trainer.fit(model_bm, train_loader)
        torch.save(model_bm.state_dict(), 'models/pretrained/' + config['train_backwardmapper']['save_name'] + '.pkl')
        

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
                                load_bm = config['train_full']['load_bm'])

        trainer = pl.Trainer(gpus=config['train_full']['gpus'], max_epochs = config['train_full']['max_epochs'])
        dataset_train = Dataset_full_model(data_dir=DATA_PATH+'train/')
        train_loader = DataLoader(dataset_train, batch_size= config['train_full']['batch_size_train'], num_workers=12)
        trainer.fit(model, train_loader)
        torch.save(model.state_dict(), 'models/pretrained/' + config['train_full']['save_name'] + '.pkl')


    else:
        print('You have to specify one of the models train_wc, train_backwardmapper, train_full')
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')   
    parser.add_argument('-m','--model', help='models are train_wc, train_backwardmapper, train_full', required=True)
    
    args = vars(parser.parse_args())
    main(args)