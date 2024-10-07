# from keras_unet_collection.models import unet_3plus_2d
from keras_unet_collection.models import unet_3plus_2d
from tqdm import tqdm
input_size = (512, 512, 3)
n_labels = 4
filter_num_down = [32, 64, 128, 256, 512]
# filter_num_skip = [32, 32, 32, 32]
# filter_num_aggregate = 160
filter_num_skip='auto'
filter_num_aggregate='auto'

stack_num_down=2
stack_num_up=1
activation='ReLU'
output_activation='Sigmoid',
batch_norm=False
pool=True
unpool=True
deep_supervision=False

backbone=None
weights='imagenet'
freeze_backbone=True
freeze_batch_norm=True
name='unet3plus'
                  
model = unet_3plus_2d(input_size, n_labels, filter_num_down)

from keras_unet_collection import losses

def hybrid_loss(y_true, y_pred):
    loss_focal = losses.focal_tversky(y_true, y_pred, alpha=0.5, gamma=4/3)
    loss_iou = losses.iou_seg(y_true, y_pred)
    
    # (x) 
    #loss_ssim = losses.ms_ssim(y_true, y_pred, max_val=1.0, filter_size=4)
    
    return loss_focal+loss_iou #+loss_ssim

from glob import glob
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  

import tensorflow as tf 
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 메모리 증가 허용
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)
from tensorflow import keras
import keras_unet_collection.utils as utils

def input_data_process(input_array):
    '''converting pixel vales to [0, 1]'''
    return input_array/255.

def target_data_process(target, num_classes):
    # '''Converting tri-mask of {1, 2, 3} to three categories.'''
    # return keras.utils.to_categorical(target_array)

    # (bs, h, w, 1) -> (bs, h, w), squeeze를 통해 채널 차원 제거
    target = np.squeeze(target, axis=-1)
    
    # (bs, h, w) -> (bs, h, w, num_classes)으로 one-hot 인코딩
    target_one_hot = keras.utils.to_categorical(target, num_classes=num_classes)
    
    return target_one_hot

input_dir = '/HDD/_projects/benchmark/semantic_segmentation/new_model/datasets/patches'
mask_input_dir = '/HDD/_projects/benchmark/semantic_segmentation/new_model/datasets/masks'

sample_names = np.array(sorted(glob(input_dir + '/*.bmp')))
label_names = np.array(sorted(glob(mask_input_dir + '/*.bmp')))

L = len(sample_names)
ind_all = utils.shuffle_ind(L)

L_train = int(0.8*L)
L_valid = int(0.1*L)
L_test = int(0.1*L)
ind_train = ind_all[:L_train]
ind_valid = ind_all[L_train:L_train+L_valid]
ind_test = ind_all[L_train+L_valid:]
print("Training:validation:testing = {}:{}:{}".format(L_train, L_valid, L_test))

valid_input = input_data_process(utils.image_to_array(sample_names[ind_valid], size=512, channel=3))
valid_target = target_data_process(utils.image_to_array(label_names[ind_valid], size=512, channel=1), n_labels)

# test_input = input_data_process(utils.image_to_array(sample_names[ind_test], size=512, channel=3))
# test_target = target_data_process(utils.image_to_array(label_names[ind_test], size=512, channel=1))

N_epoch = 30 # number of epoches
N_sample = 2 # number of samples per batch

tol = 0 # current early stopping patience
max_tol = 3 # the max-allowed early stopping patience
min_del = 0 # the lowest acceptable loss value reduction 

model.compile(loss=[hybrid_loss, hybrid_loss, hybrid_loss, hybrid_loss, hybrid_loss],
                  loss_weights=[0.25, 0.25, 0.25, 0.25, 1.0],
                  optimizer=keras.optimizers.Adam(learning_rate=1e-4))

# loop over epoches
for epoch in range(N_epoch):
    
    if epoch != 0 and epoch%30 == 0:
        model.save(f'/HDD/unet3p_{epoch}.h5')
    
    train_loss = []
    for step in range(int(L_train/N_sample)):
        print(f"\r train {str(epoch)} ({step}/{int(L_train/N_sample)}) > {str(train_loss[-1]) if len(train_loss) != 0 else ''}", end="")
        ind_train_shuffle = utils.shuffle_ind(L_train)[:N_sample]
        
        ## augmentation is not applied
        train_input = input_data_process(
            utils.image_to_array(sample_names[ind_train][ind_train_shuffle], size=512, channel=3))
        train_target = target_data_process(
            utils.image_to_array(label_names[ind_train][ind_train_shuffle], size=512, channel=1), n_labels)
        
        # train on batch
        train_loss.append(model.train_on_batch([train_input,], 
                                         [train_target, train_target, train_target, train_target, train_target,]))
        if np.isnan(train_loss[-1]):
            print("Training blow-up")
            raise Exception

    print('train loss: ', np.mean(train_loss))
       
    # epoch-end validation
    if epoch != 0 and epoch%10 == 0:
        val_loss = []
        for step in range(int(L_valid/N_sample)):
            print(f"\r val {str(epoch)} ({step}/{int(L_train/N_sample)}) > {str(train_loss[-1]) if len(train_loss) != 0 else ''}", end="")
            y_pred = model.predict([valid_input[N_sample*step:N_sample*(step + 1)]])
            val_loss.append(np.mean(hybrid_loss(valid_target[N_sample*step:N_sample*(step + 1)], y_pred)))
            
        val_loss = np.mean(val_loss)
        print('val loss: ', val_loss)
        model.save(f'/HDD/unet3p_{epoch}.h5')
            
    
    # # if loss is reduced
    # if record - record_temp > min_del:
    #     print('Validation performance is improved from {} to {}'.format(record, record_temp))
    #     record = record_temp; # update the loss record
    #     tol = 0; # refresh early stopping patience
    #     # ** model checkpoint is not stored ** #

    # # if loss not reduced
    # else:
    #     print('Validation performance {} is NOT improved'.format(record_temp))
    #     tol += 1
    #     if tol >= max_tol:
    #         print('Early stopping')
    #         break;
    #     else:
    #         # Pass to the next epoch
    #         continue;