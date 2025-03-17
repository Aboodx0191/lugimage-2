from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Input

def create_vgg16_model():
    # Explicitly define input tensor
    input_tensor = Input(shape=(224, 224, 3))
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

    # Unfreeze last 4 layers
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)  
    x = Dropout(0.5)(x)  
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output_tensor = Dense(1, activation='sigmoid')(x)  # Ensure output is a Keras tensor

    # Define the model
    model = Model(inputs=input_tensor, outputs=output_tensor)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
