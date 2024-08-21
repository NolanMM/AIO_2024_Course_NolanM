import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

Root_Dir = './Projects/Image_Retrieval/'
Class_name = sorted(os.listdir(f'{Root_Dir}/train'))


def read_image_from_path(path, size):
    im = Image.open(path).convert('RGB').resize(size)
    return np.array(im)


def folder_to_images(folder, size):
    list_dir = [f'{folder}/{name}' for name in os.listdir(folder)]
    images_np = np.zeros(shape=(len(list_dir), *size, 3))
    image_path = []
    for i, path in enumerate(list_dir):
        images_np[i] = read_image_from_path(path, size)
        image_path.append(path)
    images_path = np.array(image_path)
    return images_np, images_path


def absolute_difference(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    return np.sum(np.abs(query - data), axis=axis_batch_size)


def get_l1_score(root_image_path, query_path, size):
    query = read_image_from_path(query_path, size)
    ls_path_score = []
    for folder in os.listdir(root_image_path):
        if folder in Class_name:
            images, images_path = folder_to_images(
                f'{root_image_path}/{folder}', size)
            score = absolute_difference(query, images)
            ls_path_score.extend(list(zip(images_path, score)))

    return query, ls_path_score

root_img_path = f"./train/"
query_path = f"./test/Orange_easy/0_100.jpg"
size = (448, 448)

query, ls_path_score = get_l1_score(root_img_path, query_path, size)
