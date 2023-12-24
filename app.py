import os
from flask import Flask, render_template, request, send_file
from predictor import check
from segmentation import segmentation



app = Flask(__name__, static_folder="images")

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
@app.route('/index')
def index():
    return render_template('upload.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist('file'):
        print(file)
        filename = file.filename
        print(filename)
        dest = '/'.join([target, filename])
        print(dest)
        file.save(dest)

    # status = check(filename)
    results = check(filename)
    segmentation(filename)
    

    return render_template('complete.html', image_name=filename, results=results)


@app.route('/get_image')
def get_image():
    # target = os.path.join(APP_ROOT, 'images/')
    # if not os.path.isdir(target):
    #     os.mkdir(target)
    # for file in request.files.getlist('file'):
    #     filename = file.filename  # Replace this with your input image filename
    #     dest = '/'.join([target, filename])
    #     dest = '/'.join([target, filename])
    #     file.save(dest)
    # image_path = segmentation(filename)
    # Get the image path from the request and send the file
    image_path = 'segmentation_results/binary_mask.png'
    # image_path = request.args.get('image_path', type=str)
    return send_file(image_path, mimetype='image/png')
# @app.route('/predict_image')
# def predict_image():
#     target = os.path.join(APP_ROOT, 'images/')
#     if not os.path.isdir(target):
#         os.mkdir(target)
#     for file in request.files.getlist('file'):
#         filename = file.filename  # Replace this with your input image filename
#         dest = '/'.join([target, filename])
#         dest = '/'.join([target, filename])
#         file.save(dest)
#         output_path = segmentation(filename)  # Call the segmentation function
#     return send_file(output_path, mimetype='image/png')

# @app.route('/show_image')
# def show_image():
#     # Đường dẫn đến file ảnh bạn muốn hiển thị
#     image_path = 'testup/binary_mask.png'

#     # Trả về một trang HTML chứa ảnh
#     return render_template('complete.html', image_path=image_path)

if __name__ == "main":
    app.run(port=4555, debug=True)
