import tensorflow as tf
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.applications.vgg19 import preprocess_input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K

import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Using pretrained model VGG19 
model = VGG19(
    include_top = False,
    weights = 'imagenet')
    
model.trainable = False
model.summary()

def resize_img(img):
    ''' 
    Resizes the image to intended size.

    Arguments:
        img: Loaded image.
        
    Returns:
        img_resize: Resized Image. 
    '''
    
    img_resize = img.resize((400, 400))
    return img_resize

def load_and_process_image(image_path):
    ''' 
    Loading and preprocessing of image.
    
    Arguments: 
        image_path: Path of image.
    
    Return:
        img: Precessed image array. 
    '''
  
    img = load_img(image_path)
    img = resize_img(img)
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis = 0)
    return img

def deprocess(x):
    # perform the inverse of the preprocessing step.
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x

def display_image(image):
    if len(image.shape) == 4:
        img = np.squeeze(image, axis = 0)

    img = deprocess(img)
    
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    return

img = load_and_process_image('Content_image.jpg')
display_image(img)

img = load_and_process_image('Style_image.jpg')
display_image(img)

style_layers = [
    'block1_conv1', 
    'block3_conv1', 
    'block5_conv1'
]

content_layer = 'block5_conv2'

# Intermediate models
content_model = Model(
    inputs = model.input, 
    outputs = model.get_layer(content_layer).output
)

style_models = [Model(inputs = model.input, 
                      outputs = model.get_layer(layer).output) for layer in style_layers]

def content_cost(content, generated):
    a_C = content_model(content)
    a_G = content_model(generated)
    cost = tf.reduce_mean(tf.square(a_C - a_G))
    return cost

def gram_matrix(A):
    channels = int(A.shape[-1])
    a = tf.reshape(A, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a = True)
    return gram / tf.cast(n, tf.float32)

lam = 1. / len(style_models)

def style_cost(style, generated):
    J_style = 0
    
    for style_model in style_models:
        a_S = style_model(style)
        a_G = style_model(generated)
        GS = gram_matrix(a_S)
        GG = gram_matrix(a_G)
        current_cost = tf.reduce_mean(tf.square(GS - GG))
        J_style += current_cost * lam
    
    return J_style

generated_images = []

def training_loop(content_path, style_path, iterations = 100, a = 15., b = 25.):

    content = load_and_process_image(content_path)
    style = load_and_process_image(style_path)
    generated = tf.Variable(content)
    
    opt = tf.optimizers.Adam(learning_rate=7.)
    
    best_cost = 1e12+0.1
    best_image = None
    
    start_time = time.time()
    
    for i in range(iterations):
        
        with tf.GradientTape() as tape:
            J_content = content_cost(content, generated)
            J_style = style_cost(style, generated)
            J_total = a * J_content + b * J_style
        
        grads = tape.gradient(J_total, generated)
        opt.apply_gradients([(grads, generated)])
        
        if J_total < best_cost:
            best_cost = J_total
            best_image = generated.numpy()

        
        if i % int(iterations/10) == 0:
            time_taken = time.time() - start_time
            print('Cost at {}: {}. Time elapsed: {}'.format(i, J_total, time_taken))
            generated_images.append(generated.numpy())
        
    return best_image

with tf.device('/GPU:0'):
    final = training_loop('ContentImage_Path.jpg','StyleImage_Path.jpg')

plt.figure(figsize = (100, 100))

for i in range(10):
    plt.subplot(5, 2, i + 1)
    display_image(generated_images[i])


display_image(final)
plt.savefig('result5.jpg')
