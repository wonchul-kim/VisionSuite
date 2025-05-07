import tensorflow as tf
from tensorflow.keras import layers

def light_unet(input_shape=(512,512,3), num_classes=150):
    base = tf.keras.applications.MobileNetV2(
    input_shape=input_shape,
    include_top=False,
    weights='imagenet'
    )

    x = base.output
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)
    x = layers.UpSampling2D(8)(x) # 512x512 출력
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(x)

    return tf.keras.Model(inputs=base.input, outputs=outputs)