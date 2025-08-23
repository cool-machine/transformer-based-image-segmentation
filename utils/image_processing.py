import tensorflow as tf
from transformers import SegformerImageProcessor
import numpy as np
from PIL import Image

# Initialize SegFormer processor
image_processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
    do_rescale=False,
    do_resize=False
)

# Original 30 classes from Cityscapes
original_classes = [
    'road', 'sidewalk', 'parking', 'rail track', 'person', 'rider', 'car', 'truck', 'bus', 'on rails',
    'motorcycle', 'bicycle', 'caravan', 'trailer', 'building', 'wall', 'fence', 'guard rail', 'bridge',
    'tunnel', 'pole', 'pole group', 'traffic sign', 'traffic light', 'vegetation', 'terrain', 'sky',
    'ground', 'dynamic', 'static'
]

# Mapping to 8 major groups
class_mapping = {
    'road': 'flat', 'sidewalk': 'flat', 'parking': 'flat', 'rail track': 'flat',
    'person': 'human', 'rider': 'human',
    'car': 'vehicle', 'truck': 'vehicle', 'bus': 'vehicle', 'on rails': 'vehicle',
    'motorcycle': 'vehicle', 'bicycle': 'vehicle', 'caravan': 'vehicle', 'trailer': 'vehicle',
    'building': 'construction', 'wall': 'construction', 'fence': 'construction', 'guard rail': 'construction',
    'bridge': 'construction', 'tunnel': 'construction',
    'pole': 'object', 'pole group': 'object', 'traffic sign': 'object', 'traffic light': 'object',
    'vegetation': 'nature', 'terrain': 'nature',
    'sky': 'sky',
    'ground': 'void', 'dynamic': 'void', 'static': 'void'
}

# New labels for the 8 major groups
new_labels = {
    'flat': 0, 'human': 1, 'vehicle': 2, 'construction': 3, 'object': 4, 'nature': 5, 'sky': 6, 'void': 7
}

def process_image(image_path):
    """Process image for segmentation model input."""
    image_path = str(image_path)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image_np = image.numpy()
    image = image_processor(image_np, return_tensors="tf")['pixel_values'][0]
    image = tf.squeeze(image)
    image = tf.transpose(image, perm=[1, 2, 0])
    image = tf.image.resize(image, [512, 1024])
    image = tf.transpose(image, perm=[2, 0, 1])
    image.set_shape([3, 512, 1024])
    
    if tf.shape(image)[2] == 3:
        image = tf.transpose(image, [2, 0, 1])
    
    return image

def map_labels_tf(label_image, original_classes, class_mapping, new_labels):
    """Map original Cityscapes labels to simplified 8-class system."""
    label_image = tf.squeeze(label_image)
    label_image_shape = tf.shape(label_image)
    mapped_label_image = tf.zeros_like(label_image, dtype=tf.uint8)
    
    for original_class, new_class in class_mapping.items():
        try:
            original_class_index = tf.cast(original_classes.index(original_class), tf.uint8)
            new_class_index = tf.cast(new_labels[new_class], tf.uint8)
            mask = tf.equal(label_image, original_class_index)
            fill_val = tf.fill(label_image_shape, tf.cast(new_class_index, tf.uint8))
            mapped_label_image = tf.where(mask, fill_val, mapped_label_image)
        except Exception as e:
            tf.print(f"Error mapping class {original_class}: {e}")
            
    label = tf.expand_dims(mapped_label_image, axis=-1)
    label = tf.image.convert_image_dtype(label, tf.uint8)
    return label

def process_mask(mask_path):
    """Process ground truth mask."""
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1)
    mask = map_labels_tf(mask, original_classes, class_mapping, new_labels)
    mask = tf.image.resize(mask, [128, 256], method='nearest')
    
    if tf.shape(mask)[0] == 1:
        mask = tf.squeeze(mask, axis=0)
    if tf.shape(mask)[-1] != 1:
        mask = tf.expand_dims(mask, axis=-1)
    
    return mask