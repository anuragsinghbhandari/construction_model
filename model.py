import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib as plt
# Load the saved model
model = tf.keras.models.load_model('cnn_model.h5')

# Load and preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))  # Resize the image
    img_array = image.img_to_array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match the input shape
    img_array /= 255.0  # Normalize the image (same as during training)
    return img_array

# Make a prediction
def predict_image(img_path):
    processed_image = preprocess_image(img_path)
    predictions = model.predict(processed_image)[0]  # Get the prediction
    labels = ['green_cloth', 'parking_space', 'rainwater_harvesting', 'safety_precautioins']

    # Threshold for considering whether the category is present (e.g., > 0.5)
    result = {labels[i]: (predictions[i] > 0.5) for i in range(len(labels))}

    return result

# Display image with prediction
def show_image_with_prediction(img_path, prediction_result):
    # Load and display the image
    img = image.load_img(img_path)
    plt.imshow(img)

    # Build a string with the prediction results
    prediction_text = '\n'.join([f'{label}: {"Yes" if value else "No"}' for label, value in prediction_result.items()])

    # Add the prediction text on the image
    plt.text(10, -20, prediction_text, color='red', fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    # Show the image
    plt.axis('off')  # Hide axes
    plt.show()


def predictimg(img_path):
  prediction_result = predict_image(img_path)
  show_image_with_prediction(img_path, prediction_result)

img_path = 'path_to_new_image.jpg'
predict_image(img_path)