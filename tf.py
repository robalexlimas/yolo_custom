# all library imports
import tensorflow as tf
from numpy import expand_dims
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def tensorflow_on_gpu():
    print(tf.__version__)
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    else:
        print("Please install GPU version of TF or TF is not running on GPU")


def tensorflow_clear():
    tf.keras.backend.clear_session()


def model_load(model_path):
    model = load_model(model_path)
    model.summary()
    return model


def preprocessing_image(image_path, input_size):
    image = load_img(image_path)
    width, height = image.size
    image = load_img(image_path, target_size=(input_size[0], input_size[1]))
    image = img_to_array(image)
    image = image.astype('float32')
    image /= 255.0
    image = expand_dims(image, 0)
    return image, width, height
