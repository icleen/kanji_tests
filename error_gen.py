import os
import sys
import json

def make_html(net_name, prediction_file, dictionary_file, validation_file):
    # print(os.getcwd())
    # cwd = str(os.getcwd())
    # home_path = str('file://' + cwd)
    # print(home_path)
    root_path = '/home/iclee141/workspace/kanji_tests'
    home_path = str('file://' + root_path)

    predictions = []
    correct = []
    acc = []
    temp = []
    with open(prediction_file, 'r') as f:
        for line in f:
            temp = line.split()
            predictions.append(temp[0])
            correct.append(temp[1])
            acc.append(temp[2])

    with open(dictionary_file, 'r') as f:
        temp_dict = json.load(f)

    dictionary = {}
    for key in temp_dict:
        value = str(temp_dict[key])
        dictionary[value] = key

    with open(validation_file, 'r') as f:
        pics = json.load(f)
    # print( str('number of pics: ' + str(len(pics))) )
    # print( str('number of preds: ' + str(len(predictions))) )

    root = str(net_name + '_prediction_errors')
    if not os.path.isdir(root):
        os.makedirs(root)

    errors = []
    cors = []
    for i, pred in enumerate(predictions):
        if i < len(pics):
            img_path = pics[i]['image_path']
            cor = correct[i]
            pred_str = dictionary[pred]
            pred_utf = pred_str.split('+')[1]
            cor_str = dictionary[cor]
            cor_utf = cor_str.split('+')[1]

            file_path = str(root + '/' + str(i) + '_' + cor_str + '.html')
            full_path = str(home_path + '/' + file_path)
            if pred == cor:
                cors.append((full_path, dictionary[cor]))
            else:
                errors.append((full_path, dictionary[cor]))
            with open(file_path, 'w') as f:
                f.write('<!DOCTYPE html>\n<html>\n<body>\n')

                f.write('<p>' + img_path + '</p>')
                f.write('<img src="' + home_path + '/' + img_path + '" height="100" width="100">\n')
                f.write('<h2>Predicted: - &#x' + pred_utf + '; </h2>\n')
                f.write('<a href="http://www.fileformat.info/info/unicode/char/' + pred_utf + '/index.htm" target="_blank">' + pred_str + '</a>\n')
                # f.write('<img src="www.fileformat.info/info/unicode/char/' + pred_utf + '/sample.svg" height="100" width="100">\n')
                f.write('<h2>Ground Truth: - &#x' + cor_utf + '; ' + acc[i] + '</h2>\n')
                f.write('<a href="http://www.fileformat.info/info/unicode/char/' + cor_utf + '/index.htm" target="_blank">' + cor_str + '</a>\n')
                # f.write('<img src="www.fileformat.info/info/unicode/char/' + cor_utf + '/sample.svg" height="100" width="100">\n')

                # if pred == cor:
                #
                # if i > 0 and len(cors) > i:
                #     f.write('<a href="' + cors[i - 1][0] + '" target="_blank">back  </a>\n')
                # if i < len(cors) - 1:
                #     f.write('<a href="' + cors[i + 1][0] + '" target="_blank">  next</a>\n')

                f.write('</body>\n</html>\n')

    html_file = str(net_name + '_predictions.html')
    with open(html_file, 'w') as f:
        f.write('<!DOCTYPE html>\n<html>\n<body>\n')

        f.write('<h1>Errors</h1>\n')
        errors.sort(key=lambda x: x[1])
        for error in errors:
            f.write('<a href="' + error[0] + '">' + error[1] + '</a>\n')
            # f.write('<p></p>')

        f.write('<hr>\n')
        f.write('<h1>Correct Predictions</h1>\n')
        cors.sort(key=lambda x: x[1])
        for cor in cors:
            f.write('<a href="' + cor[0] + '">' + cor[1] + '</a>\n')
            # f.write('<p></p>')


        f.write('</body>\n</html>\n')

if __name__ == '__main__':
    prediction_file = sys.argv[1]
    dictionary_file = sys.argv[2]
    validation_file = sys.argv[3]

    make_html('test_net', prediction_file, dictionary_file, validation_file)
