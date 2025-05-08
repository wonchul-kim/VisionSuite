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

    input_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/250211/split_dataset'
    output_dir = '/HDD/etc/tensorflow'
    now = datetime.datetime.now()

    tfrecords_dir = osp.join(output_dir, 'tfrecords')
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

    epochs = 10
    global_batch_size = 64
    num_classes = 150
    roi = [220, 60, 1340, 828]
    strategy = tf.distribute.MirroredStrategy()
    print(f'활성화된 GPU 수: {strategy.num_replicas_in_sync}')


    with strategy.scope():
        height, width = 768, 1120
        base_model, layers, layer_names = create_base_model('efficientnetb3', 'imagenet', height, width)
        model = DeepLabV3plus(150, base_model, layers, height=height, width=width)
        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            1e-4, decay_steps=1000, decay_rate=0.96
        )
        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False,
                    reduction=tf.keras.losses.Reduction.NONE # 필수 변경[2][4]
                )
        train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    train_dataset = build_optimized_dataset(tfrecords_dir, global_batch_size, strategy, roi=roi)
    dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    
    train_loop(model, dist_dataset, epochs, optimizer, loss_fn, strategy, train_acc_metric)
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

    
