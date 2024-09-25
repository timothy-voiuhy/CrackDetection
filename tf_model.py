import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image
import numpy as np
import argparse

def createtfDataLoader(__dir, batch_size):
  dataset = tf.keras.utils.image_dataset_from_directory(
      __dir,
      labels = "inferred",
      label_mode = "binary",
      image_size = (128, 128),
      interpolation = "nearest",
      batch_size = batch_size,
      shuffle = True
  )
  return dataset

tf_CrackDetectionModel = Sequential([
    Conv2D(32, (3, 3), activation = "relu", input_shape = (128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation = "relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation = "relu"),
    Dense(1, activation = "sigmoid")
])

def traintfModel(model, epochs, __root = None, __train_dir=None, __test_dir = None, __val_dir = None, batch_size = 64, tf_best_model_path = None):
    if __root is not None:
        __train_dir = os.path.join(__root, __train_dir)
        __test_dir = os.path.join(__root, __test_dir)
        __val_dir = os.path.join(__root, __val_dir)
    tf_train_dataloader = createtfDataLoader(__train_dir, batch_size)
    tf_test_dataloader = createtfDataLoader(__test_dir, batch_size)
    tf_val_dataloader = createtfDataLoader(__val_dir, batch_size)

    # define model checkpoint callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath= tf_best_model_path,
        monitor = "val_accuracy",
        save_best_only = True,
        mode = 'max',
        verbose = 1
    )
    model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    model.fit(tf_train_dataloader,
              validation_data= tf_val_dataloader,
              epochs = epochs,
              callbacks = checkpoint_callback)
    model.evaluate(tf_test_dataloader)

def predict_single_image(model, image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0
    image = tf.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction[0][0]

def predict_single_image_tf(image_path, model_path = None, model = None):
    """Predicts the class of a single crack image using a TensorFlow Keras model.

    Args:
        image_path: Path to the image file.
        model_path: Path to the saved Keras model file (.keras).

    Returns:
        The predicted class (0 for no crack, 1 for crack).
    """
    if model_path is not None:
        model = tf.keras.models.load_model(model_path)
    else:
        model = model
    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Load and preprocess the image
    image = Image.open(image_path)
    image = image.resize((128, 128))  # Resize to match model input shape
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make the prediction
    prediction = model.predict(image)

    # Get the predicted class
    predicted_class = (prediction > 0.5).astype(int)[0][0]  # Assuming binary classification

    return predicted_class

def parse_args():
    parser = argparse.ArgumentParser(description="Train or predict with a TensorFlow model")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", action="store_true", help="Predict on a single image")
    parser.add_argument("--image_path", type=str, help="Path to the image file for prediction")
    parser.add_argument("--root", type=str, help="Root directory of the dataset")
    parser.add_argument("--model_path", type=str, help="Path to the saved Keras model file (.keras)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for training")
    parser.add_argument("--train_dir", type=str, default="data/train", help="Directory for training data")
    parser.add_argument("--test_dir", type=str, default="data/test", help="Directory for testing data")
    parser.add_argument("--val_dir", type=str, default="data/val", help="Directory for validation data")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and testing")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.train:
        traintfModel(tf_CrackDetectionModel, args.epochs, args.root, args.train_dir, args.test_dir, args.val_dir, args.batch_size, args.model_path)
    elif args.predict:
        predict_single_image_tf(args.image_path, args.model_path)