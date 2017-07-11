import os
import sys
import json

def make_html(prediction_file, dictionary_file, validation_file):
    print(os.getcwd())
    cwd = str(os.getcwd())
    home_path = str('file://' + cwd)
    print(home_path)

    predictions = []
    correct = []
    temp = []
    with open(prediction_file, 'r') as f:
        for line in f:
            temp = line.split()
            predictions.append(temp[0])
            correct.append(temp[1])

    with open(dictionary_file, 'r') as f:
        temp_dict = json.load(f)

    dictionary = {}
    for key in temp_dict:
        value = str(temp_dict[key])
        dictionary[value] = key

    with open(validation_file, 'r') as f:
        pics = json.load(f)


    root = 'prediction_errors'
    if not os.path.isdir(root):
        os.makedirs(root)

    errors = []
    cors = []
    for i, pic in enumerate(pics):
        img_path = pic['image_path']
        pred = predictions[i]
        cor = correct[i]
        pred_str = dictionary[pred]
        pred_utf = pred_str.split('+')[1]
        cor_str = dictionary[cor]
        cor_utf = cor_str.split('+')[1]
        file_path = str(root + '/' + str(i) + '_' + pred_str + '.html')
        full_path = str(home_path + file_path)
        if pred == cor:
            cors.append((full_path, dictionary[cor]))
        else:
            errors.append((full_path, dictionary[cor]))
        with open(file_path, 'w') as f:
            f.write('<!DOCTYPE html>\n<html>\n<body>\n')

            f.write('<img src="' + home_path + img_path + '" height="100" width="100">')
            f.write('<p>Predicted: </p>')
            f.write('<a href="http://www.fileformat.info/info/unicode/char/' + pred_utf + '/index.htm">' + pred_str + '</a>')
            f.write('<p>Ground Truth: </p>')
            f.write('<a href="http://www.fileformat.info/info/unicode/char/' + cor_utf + '/index.htm">' + cor_str + '</a>')
            f.write('<p></p>')

            f.write('</body>\n</html>\n')

    with open('predictions.html', 'w') as f:
        f.write('<!DOCTYPE html>\n<html>\n<body>\n')

        f.write('<h1>errors</h1>\n')
        for error in errors:
            f.write('<a href="' + error[0] + '">' + error[1] + '</a>\n')
            f.write('<p></p>')


        f.write('<hr>\n')
        f.write('<h1>correct predictions</h1>\n')
        for cor in cors:
            f.write('<a href="' + cor[0] + '">' + cor[1] + '</a>\n')
            f.write('<p></p>')


        f.write('</body>\n</html>\n')

if __name__ == '__main__':
    prediction_file = sys.argv[1]
    dictionary_file = sys.argv[2]
    validation_file = sys.argv[3]

    make_html(prediction_file, dictionary_file, validation_file)
