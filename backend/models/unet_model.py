import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

def create_unet_with_vgg16_encoder(input_shape=(512, 1024, 3), num_classes=8):
    """
    Create UNet model with VGG16 encoder.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of segmentation classes
        
    Returns:
        tf.keras.Model: Compiled UNet model
    """
    inputs = tf.keras.Input(input_shape)

    # Load VGG16 with pre-trained weights and exclude the top fully connected layers
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)

    # Freeze the VGG16 layers
    for layer in vgg16.layers:
        layer.trainable = False

    # Extract layers for skip connections
    skip1 = vgg16.get_layer("block1_conv2").output  # 64 filters
    skip2 = vgg16.get_layer("block2_conv2").output  # 128 filters
    skip3 = vgg16.get_layer("block3_conv3").output  # 256 filters
    skip4 = vgg16.get_layer("block4_conv3").output  # 512 filters

    # Bottleneck (last layer of VGG16)
    bottleneck = vgg16.get_layer("block5_conv3").output  # 512 filters

    # Decoder with dropout
    d1 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bottleneck)
    d1 = layers.concatenate([d1, skip4])
    d1 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(d1)
    d1 = layers.Dropout(0.5)(d1)
    d1 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(d1)

    d2 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(d1)
    d2 = layers.concatenate([d2, skip3])
    d2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(d2)
    d2 = layers.Dropout(0.5)(d2)
    d2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(d2)

    d3 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(d2)
    d3 = layers.concatenate([d3, skip2])
    d3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(d3)
    d3 = layers.Dropout(0.5)(d3)
    d3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(d3)

    d4 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(d3)
    d4 = layers.concatenate([d4, skip1])
    d4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(d4)
    d4 = layers.Dropout(0.5)(d4)
    d4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(d4)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(d4)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model