import tensorflow as tf
from transformers import AutoModel, AutoImageProcessor, SegformerConfig
from tensorflow.keras.optimizers import Adam

def create_segformer_model(num_classes=8, model_name="nvidia/segformer-b0-finetuned-cityscapes-512-1024"):
    """
    Create SegFormer model for semantic segmentation.
    
    Args:
        num_classes: Number of segmentation classes
        model_name: Pre-trained model name from HuggingFace
        
    Returns:
        TFSegformerForSemanticSegmentation: Configured model
    """
    config = SegformerConfig(
        num_labels=num_classes,
        id2label={0: "flat", 1: "human", 2: "vehicle", 3: "construction", 
                 4: "object", 5: "nature", 6: "sky", 7: "void"},
        label2id={"flat": 0, "human": 1, "vehicle": 2, "construction": 3, 
                 "object": 4, "nature": 5, "sky": 6, "void": 7},
        image_size=(512, 1024),
    )
    
    try:
        model = AutoModel.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,
            from_tf=True
        )
    except Exception as e:
        print(f"Failed to load SegFormer model: {e}")
        print("SegFormer requires PyTorch backend or specific TensorFlow version")
        return None
    
    return model

def load_segformer_checkpoint(model, checkpoint_dir="models/checkpoints/"):
    """Load model from checkpoint if available."""
    optimizer = Adam()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint).expect_partial()
        print(f"SegFormer checkpoint restored from: {latest_checkpoint}")
        return True
    else:
        print("No SegFormer checkpoint found. Model initialized from pre-trained weights.")
        return False

def predict_segformer(model, image):
    """
    Make segmentation prediction using SegFormer model.
    
    Args:
        model: SegFormer model
        image: Preprocessed image tensor
        
    Returns:
        tf.Tensor: Predicted segmentation mask
    """
    image = tf.expand_dims(image, axis=0)
    logits = model(image, training=False).logits
    
    # Transpose and get class predictions
    logits = tf.transpose(logits, perm=[0, 2, 3, 1])
    predictions = tf.argmax(logits, axis=-1)
    mask = tf.squeeze(predictions, axis=0)
    
    return mask