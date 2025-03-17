import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# Define dataset paths
train_dir = "dataset/chest_xray/train"  # Replace with actual path
val_dir = "dataset/chest_xray/val"      # Replace with actual path
test_dir = "dataset/chest_xray/test"    # Replace with actual path

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
    zoom_range=0.1, horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load Data
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=32, class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(150, 150), batch_size=32, class_mode='binary'
)
test_generator = val_datagen.flow_from_directory(
    test_dir, target_size=(150, 150), batch_size=32, class_mode='binary', shuffle=False
)

# Compute Class Weights for Imbalance
labels = train_generator.classes
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
