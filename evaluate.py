import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = tf.keras.models.load_model('best_vgg16_model.h5')  # Change filename if needed

# Load test dataset
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'dataset/chest_xray/test',  # Update with correct test directory
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Get true labels
y_true = test_generator.classes  # Actual labels

# Predict
y_pred = model.predict(test_generator)

# Convert probabilities to binary labels
if y_pred.shape[1] == 1:  # Single output neuron case
    y_pred = np.round(y_pred).astype(int).flatten()
else:  # Multi-class case (if softmax used)
    y_pred = np.argmax(y_pred, axis=1)

# Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred))

# Confusion Matrix Visualization
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
