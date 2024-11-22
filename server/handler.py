from flask import Flask, render_template, request, send_file
import os
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

# Folders
IMAGE_FOLDER = 'static/images'
RESULT_FOLDER = 'results'

# Ensure results folder exists
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

# Load YOLO model
model = YOLO('model/best.pt')

@app.route('/')
def index():
    images = os.listdir(IMAGE_FOLDER)
    images = [os.path.join(IMAGE_FOLDER, img) for img in images]
    return render_template('index.html', images=images)

@app.route('/process', methods=['POST'])
def process_image():
    image_path = request.form.get('image_path')

    if not image_path:
        return "No image selected", 400

    results = model(image_path)
    results_img = results[0].plot()
    result_img_path = os.path.join(RESULT_FOLDER, f'result_{os.path.basename(image_path)}')
    Image.fromarray(results_img).save(result_img_path)

    return send_file(result_img_path, mimetype='image/png')

# Netlify requires this as a handler
def handler(event, context):
    from flask import request
    from werkzeug.serving import run_simple
    from werkzeug.wrappers import Request, Response

    request.environ['PATH_INFO'] = event['path']
    request.environ['QUERY_STRING'] = event.get('queryStringParameters', '')

    return Response.from_app(app, environ=request.environ).get_wsgi_response()
