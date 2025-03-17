from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_vgg16_model

# Load dataset
train_dir = 'dataset/chest_xray/train'
val_dir = 'dataset/chest_xray/val'
batch_size = 32
image_size = (224, 224)  

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,     # Randomly rotate images
    width_shift_range=0.2, # Randomly shift images horizontally
    height_shift_range=0.2,# Randomly shift images vertically
    shear_range=0.2,       # Apply shearing
    zoom_range=0.2,        # Randomly zoom images
    horizontal_flip=True,  # Flip images horizontally
    fill_mode='nearest'    # Fill missing pixels
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset/chest_xray/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(val_dir, target_size=image_size, batch_size=batch_size, class_mode='binary')

# Learning rate scheduler function
def lr_schedule(epoch):
    lr = 0.001
    if epoch > 10:
        lr = 0.0001
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Define ReduceLROnPlateau callback
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,   # Reduce LR by 50%
    patience=2,   # Wait 2 epochs before reducing LR
    min_lr=1e-6   # Minimum LR value
)

# Initialize model
model = create_vgg16_model()

# Fit the model with learning rate scheduling and early stopping
model.fit(
    train_generator, 
    validation_data=val_generator, 
    epochs=15, 
    callbacks=[lr_scheduler, early_stopping, reduce_lr]
)

# Save the trained model
model.save('best_vgg16_model.h5')
