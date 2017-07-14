import os
import sys
import cv2
import json
import h5py
import codecs
import numpy as np
from scipy import ndimage, misc

def make_dictionary(folder_list, dataset_name):
    key = ''
    dictionary = {}
    for i, folder in enumerate(folder_list):
        key = os.path.basename(folder).split('/')[0]
        dictionary[key] = i
    with open('kanji_dictionary' + dataset_name + '.json', 'w') as f:
        json.dump(dictionary, f)
    return dictionary

def make_json(dataset_name):
    images_path = 'kanji_dataset/characters/'
    file_list = []
    train_list = []
    test_list = []
    val_list = []
    folder_list = []
    t = 0
    for name in os.listdir(images_path):
        folder_list.append(name)
        current = os.path.join(images_path, name)
        file_list = []
        for file in os.listdir(current):
            file_list.append(os.path.join(current, file))
        img_num = len(file_list)
        if img_num > 3:
            val_cnt = int(0.1 * img_num)
            if val_cnt < 1:
                val_cnt = 1
            test_cnt = int(0.1 * img_num)
            if test_cnt < 1:
                test_cnt = 1
            train_cnt = img_num - test_cnt - val_cnt
            train_list += file_list[:train_cnt]
            val_list += file_list[train_cnt:train_cnt+val_cnt]
            test_list += file_list[train_cnt+val_cnt:train_cnt+val_cnt+test_cnt]

    dictionary = make_dictionary(folder_list, dataset_name)

    train_list = np.array(train_list)
    np.random.shuffle(train_list)
    train_labels = []
    for file in train_list:
        gt = os.path.basename(file).split('_')[0]
        train_labels.append({"gt": dictionary[gt], "image_path": file})

    val_list = np.array(val_list)
    np.random.shuffle(val_list)
    val_labels = []
    for file in val_list:
        gt = os.path.basename(file).split('_')[0]
        val_labels.append({"gt": dictionary[gt], "image_path": file})

    test_list = np.array(test_list)
    np.random.shuffle(test_list)
    test_labels = []
    for file in test_list:
        gt = os.path.basename(file).split('_')[0]
        test_labels.append({"gt": dictionary[gt], "image_path": file})

    with open("training" + dataset_name + ".json", 'w') as f:
        json.dump(train_labels[:], f)

    with open("validation" + dataset_name + ".json", 'w') as f:
        json.dump(val_labels[:], f)

    with open("test" + dataset_name + ".json", 'w') as f:
        json.dump(test_labels[:], f)

    return len(folder_list)

# put the data into the database +++++++++++++++++++++++++++++++++++++++++
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
        img = cv2.resize(img, (size[0], size[1]),
                interpolation = cv2.INTER_LINEAR)
        train[k] = img
        labels[k] = int(obj['gt'])
    train = np.reshape(train, (len(training), size[0] * size[1]))
    train /= 255.0
    return train, labels

def data_to_base(classes, dataset_name):
    size = (32, 32)
    training, t_labels = get_data_json('training' + dataset_name + '.json', size)
    validation, v_labels = get_data_json('validation' + dataset_name + '.json', size)
    test, test_labels = get_data_json('test' + dataset_name + '.json', size)
    with h5py.File('train_val_test_data' + dataset_name, 'w') as hf:
        hf.create_dataset('classes', data = classes)
        hf.create_dataset('training', data = training[:])
        hf.create_dataset('t_labels', data = t_labels[:])
        hf.create_dataset('validation', data = validation[:])
        hf.create_dataset("v_labels", data = v_labels[:])
        hf.create_dataset('test', data = test[:])
        hf.create_dataset("test_labels", data = test_labels[:])


# get the data ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def data_from_base(data_file):
    with h5py.File(data_file,'r') as hf:
        training = np.array(hf.get('training'))
        t_labels = np.array(hf.get('t_labels'))
        validation = np.array(hf.get('validation'))
        v_labels = np.array(hf.get('v_labels'))
    return training, t_labels, validation, v_labels

def classes_from_base(data_file):
    with h5py.File(data_file,'r') as hf:
        classes = np.array(hf.get('classes'))
    return int(classes)

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

if __name__ == '__main__':
    dataset_name = '_32_distort2'
    classes = make_json(dataset_name)
    print('to json')
    data_to_base(classes, dataset_name)
