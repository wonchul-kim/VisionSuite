from visionsuite.cores.unets.keras_unet_collection.models import unet_3plus_2d

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

from visionsuite.cores.unets.keras_unet_collection import losses

def hybrid_loss(y_true, y_pred):

    loss_focal = losses.focal_tversky(y_true, y_pred, alpha=0.5, gamma=4/3)
    loss_iou = losses.iou_seg(y_true, y_pred)
    
    # (x) 
    #loss_ssim = losses.ms_ssim(y_true, y_pred, max_val=1.0, filter_size=4)
    
    return loss_focal+loss_iou #+loss_ssim