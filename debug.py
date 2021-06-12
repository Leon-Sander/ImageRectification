import pytorch_lightning as pl
from models.densenetccnl import Backwardmapper
DATA_PATH = 'Dataset Preview/Inv3D preview complete V2/data/'
data_dir= 'Dataset Preview/Inv3D preview complete V2/data/train/'


from custom_dataset import CustomImageDataset_wc
dataset_train = CustomImageDataset_wc(data_dir=DATA_PATH+'train/', transform=True)
dataset_val = CustomImageDataset_wc(data_dir=DATA_PATH+'val/', transform=True)
dataset_test = CustomImageDataset_wc(data_dir=DATA_PATH+'test/', transform=True)

from custom_dataset import Dataset_backward_mapping
train_dataset_bm = Dataset_backward_mapping(data_dir=DATA_PATH+'train/')



from torch.utils.data import DataLoader
train_loader_bm = DataLoader(train_dataset_bm, batch_size=1, num_workers=8, shuffle=True)

train_loader = DataLoader(dataset_train, batch_size= 1, num_workers=12)
val_loader = DataLoader(dataset_val, batch_size= 1, num_workers=12)
test_loader = DataLoader(dataset_test, batch_size= 1, num_workers=12)

from models import unetnc
model = unetnc.Estimator3d(input_nc=3, output_nc=3, num_downs=0)
model_bm = Backwardmapper()

#8 channel


# most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
# trainer = pl.Trainer(gpus=8) (if you have GPUs) 


#trainer = pl.Trainer(gpus=1, max_epochs = 20)



#trainer = pl.Trainer(auto_select_gpus = True, max_epochs = 100)
#trainer.fit(model, train_loader)

# Wie sieht der Batch aus, der vom trainer und train_loader generiert wird?
for batch in train_loader_bm:
   images, labels = batch

encoded, decoded = model_bm.test_forward(images)