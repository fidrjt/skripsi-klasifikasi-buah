from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import time

app = Flask(__name__)
CORS(app)

# ============================================================================
# LOAD MODEL
# ============================================================================
print("="*70)
print("LOADING MODEL...")
print("="*70)

MODEL_PATH = 'best_model_phase2.h5'

if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Model tidak ditemukan di {MODEL_PATH}")
    print("Pastikan Anda sudah download model dari Google Colab!")
    exit(1)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print(f"✓ Model loaded from: {MODEL_PATH}")
print(f"✓ Model input shape: {model.input_shape}")
print(f"✓ Model output shape: {model.output_shape}")
print("="*70)

# ============================================================================
# CLASS NAMES DAN INFORMASI BUAH
# ============================================================================

# PENTING: Urutan harus sesuai dengan training (alfabetis)!
CLASS_NAMES = ['Kupa', 'Matoa', 'Namnam']

FRUIT_INFO = {
    'Matoa': {
        'origin': 'Papua',
        'nutrition': {
            'Vitamin': 'Vitamin C, E, dan K yang tinggi untuk daya tahan tubuh',
            'Mineral': 'Kalsium dan Fosfor untuk tulang kuat',
            'Antioksidan': 'Antosianin untuk melawan radikal bebas'
        },
        'fun_facts': [
            'Matoa disebut juga "Kasai" oleh orang Papua 🌴',
            'Rasanya manis seperti campuran lengkeng dan durian! 😋',
            'Satu pohon Matoa bisa menghasilkan 500 buah dalam satu musim! 🌳'
        ],
        'coordinates': [-2.5489, 140.7148],
        'description': 'Matoa tumbuh di hutan hujan tropis Papua dan menjadi makanan favorit masyarakat lokal. Buah ini memiliki kulit coklat-ungu dan daging buah yang manis.'
    },
    'Namnam': {
        'origin': 'Maluku',
        'nutrition': {
            'Vitamin': 'Vitamin A dan C untuk mata dan kulit sehat',
            'Mineral': 'Zat Besi dan Kalium untuk darah sehat',
            'Serat': 'Serat tinggi untuk pencernaan lancar'
        },
        'fun_facts': [
            'Namnam memiliki tekstur yang unik seperti beludru 🧸',
            'Buah ini bisa dimakan langsung atau dijadikan manisan lezat! 🍬',
            'Pohon Namnam bisa hidup hingga 100 tahun! 🎂'
        ],
        'coordinates': [-3.2385, 130.1453],
        'description': 'Namnam berasal dari Maluku dan memiliki bentuk gepeng dengan warna coklat kehitaman. Buah ini sangat khas dan hanya tumbuh di Indonesia Timur.'
    },
    'Kupa': {
        'origin': 'Sulawesi',
        'nutrition': {
            'Vitamin': 'Vitamin B kompleks untuk energi',
            'Mineral': 'Magnesium dan Zinc untuk metabolisme',
            'Antioksidan': 'Flavonoid untuk kesehatan jantung'
        },
        'fun_facts': [
            'Kupa juga dikenal sebagai "Jambu Hutan" 🌲',
            'Warnanya yang merah-ungu sangat cantik! 💜',
            'Buah ini sangat disukai oleh burung-burung 🐦'
        ],
        'coordinates': [-0.9489, 119.8707],
        'description': 'Kupa berasal dari Sulawesi dan memiliki warna merah-ungu yang menarik. Buah kecil ini memiliki rasa manis segar dan tekstur halus mengkilap.'
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def preprocess_image(image_bytes):
    """
    Preprocess gambar untuk input model
    """
    try:
        # Load image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to 224x224
        img = img.resize((224, 224), Image.BILINEAR)
        
        # Convert to array
        img_array = np.array(img)
        
        # Normalize to 0-1
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

def validate_image(file):
    """
    Validasi file gambar
    """
    # Check if file exists
    if not file or file.filename == '':
        return False, "No file selected"
    
    # Check file extension
    allowed_extensions = {'jpg', 'jpeg', 'png'}
    file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    
    if file_ext not in allowed_extensions:
        return False, f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
    
    # Check file size (max 5MB)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    max_size = 5 * 1024 * 1024  # 5MB
    if file_size > max_size:
        return False, f"File too large. Max size: 5MB, your file: {file_size/(1024*1024):.2f}MB"
    
    return True, "OK"

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def home():
    """
    Home endpoint - API info
    """
    return jsonify({
        'message': 'Fruit Classification API - Buah Khas Indonesia Timur',
        'status': 'running',
        'model': 'MobileNetV2 (Transfer Learning)',
        'classes': CLASS_NAMES,
        'version': '1.0.0',
        'endpoints': {
            'GET /': 'API information',
            'GET /health': 'Health check',
            'POST /predict': 'Classify fruit image'
        }
    })

@app.route('/health')
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'timestamp': time.time()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    """
    start_time = time.time()
    
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image provided. Please upload an image.'
            }), 400
        
        file = request.files['image']
        
        # Validate file
        is_valid, message = validate_image(file)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': message
            }), 400
        
        # Read image
        image_bytes = file.read()
        
        # Preprocess
        try:
            img_array = preprocess_image(image_bytes)
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx]) * 100
        
        # Get all probabilities
        all_probabilities = {
            CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }
        
        # Get fruit info
        fruit_info = FRUIT_INFO[predicted_class]
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # milliseconds
        
        # Determine warning message
        warning = None
        if confidence < 70:
            warning = 'Tingkat kepercayaan rendah. Coba ambil foto lagi dengan pencahayaan yang lebih baik dan objek lebih jelas!'
        elif confidence < 85:
            warning = 'Tingkat kepercayaan sedang. Untuk hasil lebih akurat, pastikan pencahayaan baik dan objek tidak terhalang.'
        
        # Build response
        response = {
            'success': True,
            'predicted_class': predicted_class,
            'confidence': round(confidence, 2),
            'all_probabilities': all_probabilities,
            'fruit_info': fruit_info,
            'inference_time_ms': round(inference_time, 2),
            'warning': warning
        }
        
        # Log prediction
        print(f"\n[PREDICTION] {predicted_class} ({confidence:.2f}%) - {inference_time:.2f}ms")
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/test', methods=['GET'])
def test():
    """
    Test endpoint untuk debugging
    """
    return jsonify({
        'message': 'Test endpoint working!',
        'model_loaded': model is not None,
        'classes': CLASS_NAMES
    })

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 STARTING FLASK SERVER")
    print("="*70)
    print(f"Server running on: http://localhost:5000")
    print(f"API endpoints:")
    print(f"  - GET  /         : API info")
    print(f"  - GET  /health   : Health check")
    print(f"  - POST /predict  : Predict fruit")
    print("="*70)
    print("Press CTRL+C to stop server\n")
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )