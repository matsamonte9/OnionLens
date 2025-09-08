from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import uuid
from datetime import datetime

app = Flask(__name__)

# Max file size
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  

# Load models
onion_model = tf.keras.models.load_model('best_onion_model.keras')
armyworm_model = tf.keras.models.load_model('best_armyworm_model.h5')

# Directories
upload_dir = './images/upload/'
os.makedirs(upload_dir, exist_ok=True)

healthy_dir = './images/upload/healthy/'
unhealthy_dir = './images/upload/unhealthy/'
os.makedirs(healthy_dir, exist_ok=True)
os.makedirs(unhealthy_dir, exist_ok=True)

processed_image_dir = './images/processed/'
os.makedirs(processed_image_dir, exist_ok=True)

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    modal_message = "File is too large! Maximum upload size is 8 MB."
    return render_template('index.html', show_error_modal=True, modal_message=modal_message), 413

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS  

def preprocess_image(img_path, target_size):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return tf.convert_to_tensor(img_array, dtype=tf.float32)

def validate_onion(image):
    prediction = onion_model.predict(image)
    confidence = np.max(prediction)
    label = np.round(prediction).astype(int).flatten()[0]
    return 'Onion' if label == 1 else 'Not Onion', confidence

def detect_armyworm(image):
    prediction = armyworm_model.predict(image)
    confidence = np.max(prediction)
    label = np.argmax(prediction)
    return 'Unhealthy' if label == 1 else 'Healthy', confidence

def get_saliency_map(img_array, model, class_idx):
    with tf.GradientTape() as tape:
        tape.watch(img_array)
        predictions = model(img_array, training=False)
        loss = predictions[0, class_idx]
    grads = tape.gradient(loss, img_array)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)
    return saliency.numpy()[0]

def plot_saliency_map(img_path, saliency, save_path):
    img = plt.imread(img_path)
    plt.figure(figsize=(12, 12))  
    plt.imshow(img, alpha=0.7)  
    plt.imshow(saliency, cmap='jet', alpha=0.5)  
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')

def save_processed_image(original_image_path, save_path, classification):
    original_image = Image.open(original_image_path)
    draw = ImageDraw.Draw(original_image)
    
    try:
        font = ImageFont.truetype("arial.ttf", size=36)
    except IOError:
        font = ImageFont.load_default()
    
    text = classification
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    image_width, image_height = original_image.size
    text_position = ((image_width - text_width) // 2, image_height - text_height - 10)
    
    draw.text(text_position, text, font=font, fill=(255, 255, 255))
    original_image.save(save_path)

@app.route('/', methods=['GET'])
def home():
    return render_template('Index.html')

@app.route('/predict', methods=['POST'])
def predict():
    modal_message = ''
    show_result_modal = False
    show_error_modal = False
    processed_image_url = None
    saliency_map_url = None
    countermeasures = None  

    if 'imageUpload' not in request.files:
        modal_message = 'No file part in the request. Please upload an image.'
        show_error_modal = True
    else:
        imagefile = request.files['imageUpload']
        if imagefile.filename == '':
            modal_message = 'No selected file. Please upload an image.'
            show_error_modal = True
        elif not allowed_file(imagefile.filename):
            modal_message = 'Invalid file format. Only PNG, JPG, and JPEG are allowed.'
            show_error_modal = True
        else:
            original_filename = secure_filename(imagefile.filename)
            unique_suffix = datetime.now().strftime("%Y%m%d%H%M%S%f")  # Add a timestamp
            unique_filename = f"{uuid.uuid4().hex}_{original_filename}"  # Alternatively, use UUID
            
            image_path = os.path.join(upload_dir, unique_filename)
            imagefile.save(image_path)
            
            image = preprocess_image(image_path, target_size=(256, 256))
            
            onion_result, onion_confidence = validate_onion(image)
            
            if onion_result == 'Not Onion':
                modal_message = 'The image is not an onion. Please upload a valid onion image.'
                show_error_modal = True
            else:
                armyworm_result, armyworm_confidence = detect_armyworm(image)
                
                destination_dir = healthy_dir if armyworm_result == 'Healthy' else unhealthy_dir
                os.makedirs(destination_dir, exist_ok=True)
                destination_path = os.path.join(destination_dir, unique_filename)
                os.rename(image_path, destination_path)

                processed_image_path = os.path.join(processed_image_dir, f"processed_{unique_filename}")
                saliency_map_path = os.path.join(processed_image_dir, f"saliency_{unique_filename}")
                
                classification_text = f'Onion - {armyworm_result} ({armyworm_confidence * 100:.2f}%)'
                
                save_processed_image(destination_path, processed_image_path, classification_text)
                processed_image_url = f'/images/processed/{os.path.basename(processed_image_path)}'
                
                if armyworm_result == 'Unhealthy':
                    saliency = get_saliency_map(image, armyworm_model, class_idx=1)
                    plot_saliency_map(destination_path, saliency, saliency_map_path)
                    saliency_map_url = f'/images/processed/{os.path.basename(saliency_map_path)}'
                    
                    countermeasures = [
                        "Hand-remove and destroy the affected plants and caterpillars if the infestation is not too severe.(Alisin ng kamay at sirain ang mga apektadong halaman at uod kung hindi masyadong malala ang infestation.)",
                        "Pheromone traps should be used to catch adult moths and lower the population. Neem oil or insecticidal soap are good options for organic gardening.(Ang mga heromone traps ay dapat gamitin upang mahuli ang mga adult moth at mapababa ang populasyon. Ang neem oil o insecticidal soap ay magandang opsyon para sa organic gardening)",
                        "Clear weeds and debris from areas where armyworms could hide or lay eggs. Refrain from overwatering as this may draw bugs.(Alisin ang mga damo at mga debris mula sa mga lugar kung saan maaaring magtago o mangitlog ang mga armyworm. Iwasan ang labis na pagtutubig dahil maaari itong magdulot ng mga insekto.)",
                        "Keep an eye out for early indications of infestation in plants. Introduce armyworm-eating birds or parasitic wasps as natural predators.(Bantayan ang mga maagang indikasyon ng infestation sa mga halaman. Ipakilala ang mga ibong kumakain ng armyworm o parasitic wasps bilang natural na mga mandaragit.)",
                        "Contact your regional agricultural extension services for specialized guidance and assistance.(Makipag-ugnayan sa iyong panrehiyong serbisyo sa pagpapalawig ng agrikultura para sa espesyal na patnubay at tulong.)"
                    ]
                
                show_result_modal = True
    
    return render_template('Index.html',
                           show_result_modal=show_result_modal,
                           show_error_modal=show_error_modal,
                           modal_message=modal_message,
                           processed_image_url=processed_image_url,
                           saliency_map_url=saliency_map_url,
                           countermeasures=countermeasures)

@app.route('/images/processed/<filename>')
def send_processed_image(filename):
    return send_from_directory(processed_image_dir, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
