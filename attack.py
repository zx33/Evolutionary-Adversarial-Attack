import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
import keras
import random
import matplotlib as mpl
mpl.use('Agg')

import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

from EA_Util import EA_Util

PATCH_SIZE = 14
BLOCK_CNT = 16
GEN_SIZE = BLOCK_CNT * BLOCK_CNT
IMAGE_URL = './Afghan_hound.jpeg'

def get_mask(indicator):
    mask = np.ones(shape=(224, 224), dtype=np.int32)
    for i in range(224):
        for j in range(224):
            mask[i][j] = indicator[i // PATCH_SIZE * BLOCK_CNT + j // PATCH_SIZE]
    return mask

def wp_eval_func(net, img, ori_label):
    def eval_func(individual):
        x = image.img_to_array(img)
        mask = get_mask(individual)
        mask = np.array([mask, mask, mask])
        mask = mask.transpose(1, 2, 0)
        x *= mask
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        tmp_pred = net.predict(x)
        if np.argmax(tmp_pred[0]) == ori_label:
            fitness = -1
        else:
            fitness = 1
        fitness *= np.sum(individual)
        fitness += (1.0 - tmp_pred[0][ori_label])
        return fitness
    return eval_func

def original_class(net, img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    ori_preds = net.predict(x)
    max_class = np.argmax(ori_preds)
    return max_class, ori_preds[0][max_class]

predict_model = ResNet50(weights='imagenet')
img = image.load_img(IMAGE_URL, target_size=(224, 224))

ori_label, ori_logits = original_class(predict_model, img)
print('Target Class: %d; Logits: %.6f' % (ori_label, ori_logits))

ea_helper = EA_Util(GEN_SIZE, eval_func=wp_eval_func(predict_model, img, ori_label), max_gen=50)
elite = ea_helper.evolution()
elite = ea_helper.population[elite]
print(elite)
mask = get_mask(elite)
mask = np.array([mask, mask, mask])
mask = mask.transpose(1, 2, 0)
x = image.img_to_array(img)
x *= mask
masked_img = image.array_to_img(x)

plt.figure()
plt.axis('off')
plt.imshow(masked_img)
plt.savefig('masked.png')

plt.figure()
plt.axis('off')
plt.imshow(img)
plt.savefig('original.png')