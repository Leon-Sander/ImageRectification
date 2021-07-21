## Image Rectification

#### Setup
Install the requirements.

#### Train models

There are three models:
 - unetnc.Estimator3d
 - backwardmapper.Backwardmapper
 - full_model.crease

To train a model you have to specify the necessary parameters and hyperparameters within the **config/train_config.json** file.

Then run: **python3 train.py --model $MODEL**
replace $MODEL with
 - train_wc
 - train_backwardmapper
 - train_full

The model will be saved in the directory models/pretrained/$SAVE_NAME.pkl
The save_name can bespecified within the config file.

#### Config file
 
 Specify the hyperparameters for the training,
 the parameters for the model,
 the data path where the data lies,
 the name under which the trained model should be saved.

 todo:
 - utils anpassen
 - unwarping anpassen bzgl eingabeformat