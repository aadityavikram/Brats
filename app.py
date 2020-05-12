from os.path import join, exists
from os import makedirs, listdir, remove
from flask import (Flask, redirect, render_template, request)

from Final_Year_Project.test_app import main

app = Flask('Brain Tumor Detection and Segmentation')
app.secret_key = 'cykablyat'
temp_data_path = 'static/data/data/'
temp_result_path = 'static/data/result/'


@app.route('/<path:path>')
def static_file(path):
    return app.send_static_file(path)


# force browser to hold no cache. Otherwise old result might return.
@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


def create_temp_dirs():
    # creating temporary data and result folders
    if not exists(temp_data_path):
        makedirs(temp_data_path)
    else:
        for files in listdir(temp_data_path):
            remove(join(temp_data_path, files))

    if not exists(temp_result_path):
        makedirs(temp_result_path)
    else:
        for files in listdir(temp_result_path):
            remove(join(temp_result_path, files))


# main flow of program
@app.route('/', methods=['GET', 'POST'])
def upload_npz():
    create_temp_dirs()
    if request.method == 'POST':
        content_file = request.files['npz-file']
        content_name = content_file.filename
        content_path = join(temp_data_path, content_name)

        if '.' in content_name and content_name.rsplit('.', 1)[1].lower() == 'npz':
            content_file.save(content_path)

        img_paths = ['1_original_data.png', '2_overall_prediction.png', '3_final_prediction.png',
                     'flair.png', 't1ce.png', 't1.png', 't2.png',
                     'overall.png', 'core.png', 'et.png', 'full.png']

        msg = main([content_name])
        include = False
        for i in range(7, 11):
            if exists(temp_result_path + img_paths[i]):
                include = True
                break
        if include:
            params_tumor = {
                'content': temp_data_path + content_name,
                'message': msg[0],
                'flair': temp_result_path + img_paths[3],
                'overall': temp_result_path + img_paths[7],
                'core': temp_result_path + img_paths[8],
                'et': temp_result_path + img_paths[9],
                'full': temp_result_path + img_paths[10]
            }
            return render_template('tumor.html', **params_tumor)
        else:
            params_non_tumor = {
                'content': temp_data_path + content_name,
                'message': msg[0],
                'flair': temp_result_path + img_paths[3],
                't1ce': temp_result_path + img_paths[4],
                't1': temp_result_path + img_paths[5],
                't2': temp_result_path + img_paths[6]
            }
            return render_template('non_tumor.html', **params_non_tumor)
    return render_template('home.html')


@app.route('/img', methods=['GET', 'POST'])
def upload_nii():
    create_temp_dirs()
    if request.method == 'POST':
        get_files = ['flair-file', 't1ce-file', 't1-file', 't2-file']
        names_list = []

        for item in get_files:
            file_got = request.files[item]
            file_name = file_got.filename
            file_path = join(temp_data_path, file_name)
            names_list.append(file_name)
            if '.' in file_name and file_name.rsplit('.', 1)[1].lower() == 'nii':
                file_got.save(file_path)

        img_paths = ['1_original_data.png', '2_overall_prediction.png', '3_final_prediction.png',
                     'flair.png', 't1ce.png', 't1.png', 't2.png',
                     'overall.png', 'core.png', 'et.png', 'full.png']

        msg = main(names_list)
        include = False
        for i in range(7, 11):
            if exists(temp_result_path + img_paths[i]):
                include = True
                break
        if include:
            params_tumor = {
                'message': msg[0],
                'flair': temp_result_path + img_paths[3],
                'overall': temp_result_path + img_paths[7],
                'core': temp_result_path + img_paths[8],
                'et': temp_result_path + img_paths[9],
                'full': temp_result_path + img_paths[10]
            }
            return render_template('tumor.html', **params_tumor)
        else:
            params_non_tumor = {
                'message': msg[0],
                'flair': temp_result_path + img_paths[3],
                't1ce': temp_result_path + img_paths[4],
                't1': temp_result_path + img_paths[5],
                't2': temp_result_path + img_paths[6]
            }
            return render_template('non_tumor.html', **params_non_tumor)
    return render_template('home.html')


@app.errorhandler(404)
def page_not_found(error):
    return render_template('page_not_found.html'), 404


if __name__ == "__main__":
    app.run(host='localhost', port=5000, debug=False, threaded=False)
