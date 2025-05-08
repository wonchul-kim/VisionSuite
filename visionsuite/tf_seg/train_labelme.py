import tensorflow as tf
import os.path as osp
import datetime
import os

if __name__ == "__main__":
    from loops.epoch_based_loop import train_loop
    from datasets.labelme import labelme2tfrecord_auto_shard, build_optimized_dataset
    from losses import categorical_crossentropy
    from models.deeplabv3plus import DeepLabV3plus
    from models.backbones import create_base_model

    input_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/250211/split_dataset_unit'
    output_dir = '/HDD/etc/tensorflow'
    now = datetime.datetime.now()

    tfrecords_dir = osp.join(output_dir, 'tfrecords')
    '''
        >>>>> time for epoch:  68.8586437702179
        0/10 > 99: loss(1199703.5000), gpu(13.49GB)>>>>> time for epoch:  51.65077018737793
        0/10 > 199: loss(1194212.1250), gpu(24.53GB)>>>>> time for epoch:  44.69721555709839
        0/10 > 238: loss(1193024.0000), gpu(14.90GB)
        Epoch 1, Loss: 1216328.3750, Accuracy: 0.2569
        EPOCH is done:  173.96172833442688
        >>>>> time for epoch:  0.48888683319091797
        1/10 > 99: loss(1190696.2500), gpu(24.45GB)>>>>> time for epoch:  22.676891565322876
        1/10 > 199: loss(1188137.3750), gpu(23.54GB)>>>>> time for epoch:  22.758178234100342
        1/10 > 238: loss(1187881.7500), gpu(16.67GB)
        Epoch 2, Loss: 1189861.6250, Accuracy: 0.2940
        EPOCH is done:  54.536705493927
        >>>>> time for epoch:  0.5064129829406738
        2/10 > 99: loss(1185724.7500), gpu(20.13GB)>>>>> time for epoch:  22.746150255203247
        2/10 > 199: loss(1184754.0000), gpu(23.49GB)>>>>> time for epoch:  22.701698064804077
        2/10 > 238: loss(1183944.0000), gpu(13.50GB)
        Epoch 3, Loss: 1185543.0000, Accuracy: 0.3481
        EPOCH is done:  54.52851676940918
        >>>>> time for epoch:  0.5280563831329346
        3/10 > 99: loss(1182047.7500), gpu(23.13GB)>>>>> time for epoch:  22.74532175064087
        3/10 > 199: loss(1180976.5000), gpu(23.01GB)>>>>> time for epoch:  22.752564668655396
        3/10 > 238: loss(1180538.7500), gpu(16.40GB)
        Epoch 4, Loss: 1182005.0000, Accuracy: 0.4181
        EPOCH is done:  54.70700478553772
    '''
    
    if not osp.exists(tfrecords_dir):
        os.mkdir(tfrecords_dir)
            
        labelme2tfrecord_auto_shard(
            data_root=input_dir,
            output_dir=tfrecords_dir,
            split='train',
            class_names = ['MARK', 'CHAMFER_MARK', 'LINE']
        )

    output_dir = osp.join(output_dir, f'{now.year}{now.month}{now.day}-{now.hour}{now.minute}{now.second}')
    logs_dir = osp.join(output_dir, 'logs')
    
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    if not osp.exists(logs_dir):
        os.mkdir(logs_dir)

    epochs = 3
    batch_size = 16
    num_classes = 4
    roi = [220, 60, 1340, 828]
    strategy = tf.distribute.MirroredStrategy()
    print(f'활성화된 GPU 수: {strategy.num_replicas_in_sync}')


    with strategy.scope():
        height, width = 768, 1120
        '''
        
            effb0
            Total params: 10,783,535
            Trainable params: 0
            Non-trainable params: 10,783,535
        
            tf - deeplabv3plus
            Total params: 2,329,209
            Trainable params: 2,139,820
            Non-trainable params: 189,389
            
            
            torch - deeplabv3plus
            Total params: 18,352,180
            Trainable params: 18,352,180
            Non-trainable params: 0
            ----------------------------------------------------------------
            Input size (MB): 9.84
            Forward/backward pass size (MB): 54425825554351.47
            Params size (MB): 70.01
            Estimated Total Size (MB): 54425825554431.32
            
            
            torch - efficientnetb0
            Total params: 5,858,704
            Trainable params: 5,858,704
            Non-trainable params: 0
            ----------------------------------------------------------------
            Input size (MB): 9.84
            Forward/backward pass size (MB): 2697.39
            Params size (MB): 22.35
            Estimated Total Size (MB): 2729.58

        '''
        base_model, layers, layer_names = create_base_model('efficientnetb3', 'imagenet', height, width)
        model = DeepLabV3plus(num_classes, base_model, layers, height=height, width=width)
        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            1e-4, decay_steps=1000, decay_rate=0.96
        )
        
        
        
        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False,
                    reduction=tf.keras.losses.Reduction.NONE # 필수 변경[2][4]
                )
        train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        
    

    # train_dataset = build_optimized_dataset(tfrecords_dir, batch_size, strategy, roi=roi)
    # dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    
    # train_loop(model, dist_dataset, epochs, optimizer, loss_fn, strategy, train_acc_metric)
    # import cv2 
    # import numpy as np
    # sample_batch = next(iter(dist_dataset))
    # per_replica_images = strategy.experimental_local_results(sample_batch[0])
    # print("replica 0 image shape:", per_replica_images[0].shape) 
    # cv2.imwrite("/HDD/etc/image0.png", per_replica_images[0][0].numpy())
    # per_replica_masks = strategy.experimental_local_results(sample_batch[1])
    # print("replica 0 mask shape:", per_replica_masks[0].shape)
    # print("replica 0 mask unique:", np.unique(per_replica_masks[0][0].numpy()))
    # cv2.imwrite("/HDD/etc/mask0.png", per_replica_masks[0][0].numpy()*60)
    # print("replica 0 mask unique:", np.unique(per_replica_masks[0][0].numpy()))
    # per_replica_filename = strategy.experimental_local_results(sample_batch[2])
    # print("replica 0 filename:", per_replica_filename[0][0])


    # per_replica_images = strategy.experimental_local_results(sample_batch[0])
    # print("replica 1 image shape:", per_replica_images[0].shape) 
    # cv2.imwrite("/HDD/etc/image1.png", per_replica_images[0][1].numpy())
    # per_replica_masks = strategy.experimental_local_results(sample_batch[1])
    # print("replica 1 mask shape:", per_replica_masks[0].shape)
    # print("replica 1 mask unique:", np.unique(per_replica_masks[0][1].numpy()))
    # cv2.imwrite("/HDD/etc/mask1.png", per_replica_masks[0][1].numpy()*60)
    # print("replica 1 mask unique:", np.unique(per_replica_masks[0][1].numpy()))
    # per_replica_filename = strategy.experimental_local_results(sample_batch[2])
    # print("replica 1 filename:", per_replica_filename[0][1])

    
