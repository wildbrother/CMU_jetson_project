# from face_recognition import FaceRecognition

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, roc_auc_score, accuracy_score

import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import random
import numpy as np
import cv2
import base64
from tqdm import tqdm
import requests
from pprint import pprint
import PIL.Image
from PIL import Image, ImageDraw, ImageFont
from numpy import expand_dims

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import pandas as pd

# workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.9, post_process=True,
    device=device, keep_all=True
)

model = InceptionResnetV1(pretrained='vggface2').eval()
print('Loaded Model')

#######################################
########### Getting Dataset ###########
#######################################

ROOT_FOLDER ="./lfw-deepfunneled/lfw-deepfunneled/"

dataset = []
for path in glob.iglob(os.path.join(ROOT_FOLDER, "**", "*.jpg")):
    person = path.split("/")[-2]
    dataset.append({"person":person, "path": path})
    
dataset = pd.DataFrame(dataset)
dataset = dataset.groupby("person").filter(lambda x: len(x) > 10)
# print(dataset.head(10))

train, test = train_test_split(dataset, test_size=0.1, random_state=0)
print("Train:", len(train))
print("Test:", len(test))

def load_image_file(file, mode='RGB'):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array
    :param file: image file name or file object to load
    :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
    :return: image contents as numpy array
    """
    im = PIL.Image.open(file)
    # im = im.resize((160, 160))
    if mode:
        im = im.convert(mode)
    return np.array(im)

X_train, y_train, X_test, y_test = list(), list(), list(), list()

### train dataset
for i in tqdm(range(len(train))):
    face_pixels = load_image_file(train['path'].iloc[i])
    img = mtcnn(face_pixels)
    face_encoding = model(img[0].unsqueeze(0)).detach().numpy()
    X_train.append(img)
    y_train.append(train['person'].iloc[i])

print("Successfully save train dataset!")

with open('./lfw_pickle/X_train_LFW.pickle', 'wb') as f:
    pickle.dump(X_train, f, pickle.HIGHEST_PROTOCOL)

with open('./lfw_pickle/y_train_LFW.pickle', 'wb') as f:
    pickle.dump(y_train, f, pickle.HIGHEST_PROTOCOL)

### test dataset
for i in tqdm(range(len(test))):
    face_pixels = load_image_file(test['path'].iloc[i])
    img = mtcnn(face_pixels)
    # face_encoding = model(img[0].unsqueeze(0)).detach().numpy()
    X_test.append(img)
    y_test.append(test['person'].iloc[i])

print("Successfully save test dataset!")

with open('./lfw_pickle/X_test_LFW.pickle', 'wb') as f:
    pickle.dump(X_test, f, pickle.HIGHEST_PROTOCOL)

with open('./lfw_pickle/y_test_LFW.pickle', 'wb') as f:
    pickle.dump(y_test, f, pickle.HIGHEST_PROTOCOL)



# ROOT_FOLDER ="./lfw-deepfunneled/lfw-deepfunneled/"


# with open('./lfw_pickle/X_train_LFW.pickle', 'rb') as f:
#     X_train = pickle.load(f)

# with open('./lfw_pickle/y_train_LFW.pickle', 'rb') as f:
#     y_train = pickle.load(f)

# print("Successfully bring train dataset!")


# with open('./lfw_pickle/X_test_LFW.pickle', 'wb') as f:
#     X_test = pickle.load(f)

# with open('./lfw_pickle/y_test_LFW.pickle', 'wb') as f:
#     y_test = pickle.load(f)

# print("Successfully bring test dataset!")