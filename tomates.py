import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Specify the directory containing your images
directory = 'tomato'

# Load the dataset
dataset = image_dataset_from_directory(
    directory,
    labels='inferred',
    label_mode='binary',  # Assuming binary classification between healthy and unhealthy
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 256),  # Adjust the image size as needed
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    data_format=None,
    verbose=True,
)

# Now you can iterate through the dataset to access your images and labels
for images, labels in dataset:
    print(images.shape, labels.shape)
    break  # Just to check the shape, remove this line after checking
