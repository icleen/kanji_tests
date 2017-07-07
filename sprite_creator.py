import os
import sys
import cv2
import json
import h5py
import codecs
import numpy as np
from scipy import ndimage, misc
from PIL import Image, ImageDraw, ImageFont
import glob

def make_sprites(json_path, img_size, number_per_folder):
    with open(json_path, 'r') as f:
        dataset = json.load(f)

    size = img_size
    string = str(str(size) + ' ')

    length = len(dataset)
    images = np.ndarray(shape=(length, size, size), dtype=np.float32)
    img = []
    labels = []
    seperater = -1
    for k, obj in enumerate(dataset):
        if k % number_per_folder == 0:
            seperater += 1

        image_path = obj['image_path']
        # print(image_path)
        img = cv2.imread(image_path,1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (size, size), interpolation = cv2.INTER_LINEAR )

        # new_path = str('sprites/sprite_' + str(k) + '.jpg')
        base_path = str('sprites' + str(img_size) + '/' + str(seperater) + '/')
        new_path = str(base_path + str(k) + '_' + str(obj['gt']) + '.jpg')
        if not os.path.exists(os.path.join(base_path)):
            os.makedirs(base_path)
        cv2.imwrite(new_path, img)
        # if not os.path.exists(os.path.join(new_path)):
        #     print('doesnt exist')
        images[k] = img
        labels.append(obj['gt'])

    with open('sprites' + str(img_size) + '/labels.txt', 'w') as f:
        for i, label in enumerate(labels):
            string = str(str(i) + '_' + str(label) + '\n')
            f.write(string)

def make_one_sprite(index, number_per_folder, img_size, folder, subdir, sprite_img, save_path):
    #get your images using glob
    iconMap = glob.glob(folder + '/' + subdir + '/*.jpg')
    iconMap = sorted(iconMap)

    print (len(iconMap))
    images = [Image.open(filename) for filename in iconMap]
    print ("%d images will be combined" % len(images))
    assert(len(images) <= number_per_folder)

    x = 0
    y = index
    image_per_row = int(number_per_folder)
    for count, image in enumerate(images):
        # x = count % image_per_row
        locationx = count * img_size
        locationy = y * img_size
        sprite_img.paste(image,(locationx,locationy))
    print ("done adding icons")
    print ('saving ' + save_path + '...')
    sprite_img.save(save_path, transparency=0)
    print ("saved")
    return sprite_img


def make_the_sprite(size):
    print ("creating image...")
    master = Image.new(
        mode='RGBA',
        size=(size, size),
        color=(0,0,0,0))  # fully transparent

    print ("created")
    return master

if __name__ == "__main__":
    json_path = sys.argv[1]
    img_size = 20
    number_per_folder = 80
    sqrt_number = 30
    make_sprites(json_path, img_size, number_per_folder)
    folder_count = -1
    sprite_folder = str('sprites' + str(img_size))

    # subdirs = []
    num_subdirs = 0
    for name in os.listdir(sprite_folder):
        if os.path.isdir(os.path.join(sprite_folder, name)):
            # subdirs.append(name)
            num_subdirs += 1

    sprite_img = make_the_sprite(number_per_folder * img_size)
    save_path = str('sprites' + str(img_size) + '/master.jpg')
    # subdirs.sort()
    for i in range(num_subdirs):
        print(sprite_folder + '/' + str(i))
        sprite_img = make_one_sprite(i, number_per_folder, img_size, sprite_folder, str(i), sprite_img, save_path)









    # the end
