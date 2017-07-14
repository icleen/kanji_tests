import os
import sys
import cv2
import json
import h5py
import codecs
import numpy as np
from scipy import ndimage, misc
from scipy.interpolate import griddata

def make_distortion(image_path):
    img = cv2.imread(image_path,1)

    # random_state = np.random.RandomState(input_data.get("random_seed", 0))
    random_state = np.random.RandomState()

    # w_mesh_interval = kwargs.get('w_mesh_interval', 25)
    w_mesh_interval = 15
    # w_mesh_variance = kwargs.get('w_mesh_variance', 3.0)
    w_mesh_variance = 2.0

    # h_mesh_interval = kwargs.get('h_mesh_interval', 25)
    h_mesh_interval = 15
    # h_mesh_variance = kwargs.get('h_mesh_variance', 3.0)
    h_mesh_variance = 2.0

    # width_variance = kwargs.get('width_variance', 0.2)
    width_variance = 0.2

    # width_center_variance = kwargs.get('width_center_variance', 0.25)
    width_center_variance = 0.25
    # height_center_variance = kwargs.get('height_center_variance', 0.25)
    height_center_variance = 0.25

    h, w = img.shape[:2]
    ###### Make it so that it is an even interval close to the requested ######
    w_ratio = w / float(w_mesh_interval)
    h_ratio = h / float(h_mesh_interval)

    w_ratio = max(1, round(w_ratio))
    h_ratio = max(1, round(h_ratio))

    w_mesh_interval = w / w_ratio
    h_mesh_interval = h / h_ratio
    ###########################################################################

    # w_scale_factor = (np.random.random() * 0.25) + 1.0
    # w_scale_factor = (np.random.random() * 0.25) + 1.0 - 0.25 / 2.0
    # w_scale_factor = min(1.0, w_scale_factor)
    w_scale_factor = (np.random.random() * 0.25) + 1.0 - 0.25 / 2.0

    c_i = w/2 - random_state.normal(0.25, height_center_variance) * w

    h_scale_factor = 1.0
    c_j = h/2 - random_state.normal(0.25, width_center_variance) * h

    new_width = int(w_scale_factor * w)
    grid_x, grid_y = np.mgrid[0:h, 0:new_width]

    source = []
    for i in np.arange(0, h+0.0001, h_mesh_interval):
        # print i
        for j in np.arange(0, w+0.0001, w_mesh_interval):
            source.append((i,j))

    destination = []
    for i, j in source:

        r_i, r_j = i, j

        r_i = random_state.normal(r_i, h_mesh_variance)
        r_i = h_scale_factor * r_i + c_i - h_scale_factor * c_i

        # Don't use the center because it makes sense that the word starts
        # at the beginning of the line
        r_j = random_state.normal(r_j, w_mesh_variance)
        r_j = w_scale_factor * r_j

        destination.append((r_i, r_j))
    grid_z = griddata(destination, source, (grid_x, grid_y), method='linear')
    map_x = np.append([], [ar[:,1] for ar in grid_z]).reshape(h,new_width)
    map_y = np.append([], [ar[:,0] for ar in grid_z]).reshape(h,new_width)
    map_x_32 = map_x.astype('float32')
    map_y_32 = map_y.astype('float32')

    warped = cv2.remap(img, map_x_32, map_y_32, cv2.INTER_LINEAR, borderValue=(255,255,255))

    return warped

def make_images_even():
    images_path = 'kanji_dataset/characters/'

    folder_list = []
    minimum = 100
    t = 0
    for name in os.listdir(images_path):
        current = os.path.join(images_path, name)
        file_list = []
        for file in os.listdir(current):
            file_list.append(os.path.join(current, file))
        img_num = len(file_list)

        if img_num > 3 and img_num < minimum:
            num_to_make = minimum - img_num
            for i in range(num_to_make):
                img = make_distortion(file_list[i % img_num])
                new_path = str(current + '/' + file_list[i%img_num].split('/')[-1].split('_')[0] + '_distorted2_' + str(i) + '.jpg')
                cv2.imwrite(new_path, img)

        # if t < 20:
        #     print(current)
        #     if img_num > 3 and img_num < minimum:
        #         num_to_make = minimum - img_num
        #         print(img_num)
        #         print(num_to_make)
        #         for i in range(num_to_make):
        #             img = make_distortion(file_list[i % img_num])
        #             # new_path = str(current + '/' + file_list[i%img_num].split('/')[-1].split('_')[0] + '_distorted_' + str(i) + '.jpg')
        #             new_path = str('distorted/' + file_list[i%img_num].split('/')[-1].split('_')[0] + '_distorted_' + str(i) + '.jpg')
        #             print(new_path)
        #             cv2.imwrite(new_path, img)
        #
        # t += 1


if __name__ == "__main__":
    make_images_even()
