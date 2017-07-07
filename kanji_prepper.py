import os
import sys
import cv2
import json
import h5py
import codecs
import numpy as np
from scipy import ndimage, misc

def make_dictionary(folder_list):
    key = ''
    dictionary = {}
    for i, folder in enumerate(folder_list):
        key = os.path.basename(folder).split('/')[0]
        dictionary[key] = i

    return dictionary

def make_json():
    image_path = 'kanji_dataset/characters/'
    file_list = []
    folder_list = []
    for subdir, dirs, files in os.walk(image_path):
        folder_list.append(subdir)
        for file in files:
            file_path = os.path.join(subdir,file)
            file_list.append(file_path)

    dictionary = make_dictionary(folder_list)
    file_list =  np.array(file_list)
    np.random.shuffle(file_list)
    labels = []
    for file in file_list:
        gt = os.path.basename(file).split('_')[0]
        labels.append({"gt": dictionary[gt], "image_path": file})

    number_of_examples = len(file_list)
    print(number_of_examples)
    train_cnt = int(number_of_examples * 0.8)
    print(train_cnt)
    val_cnt = number_of_examples - train_cnt
    print(val_cnt)
    test_cnt = int(val_cnt * 0.5)
    print(test_cnt)
    test_cnt += train_cnt
    print(test_cnt)

    with open("training.json", 'w') as f:
        json.dump(labels[:train_cnt], f)

    with open("validation.json", 'w') as f:
        json.dump(labels[train_cnt:test_cnt], f)

    with open("test.json", 'w') as f:
        json.dump(labels[test_cnt:], f)

def data_to_base():
    size = (32, 32)
    training, t_labels = get_data_json('training.json', size)
    validation, v_labels = get_data_json('validation.json', size)
    test, test_labels = get_data_json('test.json', size)
    with h5py.File('train_val_test_data_32', 'w') as hf:
         hf.create_dataset('training', data = training[:])
         hf.create_dataset('t_labels', data = t_labels[:])
         hf.create_dataset('validation', data = validation[:])
         hf.create_dataset("v_labels", data = v_labels[:])
         hf.create_dataset('test', data = test[:])
         hf.create_dataset("test_labels", data = test_labels[:])

def data_from_base(data_file):
    with h5py.File(data_file,'r') as hf:
        training = np.array(hf.get('training'))
        t_labels = np.array(hf.get('t_labels'))
        validation = np.array(hf.get('validation'))
        v_labels = np.array(hf.get('v_labels'))
    return training, t_labels, validation, v_labels

def training_from_base(data_file):
    with h5py.File(data_file,'r') as hf:
        training = np.array(hf.get('training'))
        t_labels = np.array(hf.get('t_labels'))
    return training, t_labels

def validation_from_base(data_file):
    with h5py.File(data_file,'r') as hf:
        validation = np.array(hf.get('validation'))
        v_labels = np.array(hf.get('v_labels'))
    return validation, v_labels

def test_from_base(data_file):
    with h5py.File(data_file,'r') as hf:
        test = np.array(hf.get('test'))
        test_labels = np.array(hf.get('test_labels'))
    return test, test_labels

def get_data_json(file_path, size):
    root_path = ''
    with open(file_path, 'r') as f:
        training = json.load(f)

    train = np.ndarray(shape=(len(training), size[0], size[1]), dtype=np.float32)
    img = []
    labels = np.ndarray(shape=len(training), dtype=np.int32)
    for k, obj in enumerate(training):
        image_path = root_path + obj['image_path']
        img = cv2.imread(image_path,1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (size[0], size[1]), interpolation = cv2.INTER_LINEAR )
        train[k] = img
        labels[k] = int(obj['gt'])
    train = np.reshape(train, (len(training), size[0] * size[1]))
    train /= 255.0
    return train, labels



if __name__ == '__main__':
    make_json()
    print('to json')
    data_to_base()
