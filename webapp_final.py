"""
Simple Flask Web App for Pneumonia Detection
"""
from flask import Flask, render_template, request, jsonify, Response
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image
import numpy as np
import os
from datetime import datetime
import csv
from io import StringIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
scan_history = []

# Create uploads folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model at startup
print("[INFO] Loading model...")
MODEL_PATH = "model2result.keras"
model = tf.keras.models.load_model(MODEL_PATH)
print("[OK] Model loaded successfully")

def predict_pneumonia(image_path):
    """Predict if X-ray shows pneumonia"""
    try:
        # Load and preprocess image (same as training)
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)[0][0]
        
        # Interpret result (0 = NORMAL, 1 = PNEUMONIA)
        if prediction > 0.5:
            result = "⚠️ PNEUMONIA DETECTED"
            confidence = float(prediction * 100)
            status = "warning"
        else:
            result = "✅ NORMAL - No Pneumonia"
            confidence = float((1 - prediction) * 100)
            status = "success"
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")
    
    return {
        'result': result,
        'confidence': round(confidence, 1),
        'status': status,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Only image files allowed (PNG, JPG, JPEG)'}), 400
    
    patient_id = request.form.get('patient_id', '').strip()
    if not patient_id:
        return jsonify({'error': 'Patient ID is required'}), 400

    try:
        # Save uploaded file
        filename = f"xray_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result = predict_pneumonia(filepath)

        # Store in history for CSV export
        scan_history.append({
            'patient_id': patient_id,
            'result': result['result'],
            'confidence': result['confidence'],
            'status': result['status'],
            'timestamp': result['timestamp']
        })
        
        # Clean up
        os.remove(filepath)
        
        result['patient_id'] = patient_id
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/download_csv', methods=['GET'])
def download_csv():
    """Download prediction history as CSV"""
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['patient_id', 'result', 'confidence', 'status', 'timestamp'])
    for entry in scan_history:
        writer.writerow([
            entry['patient_id'],
            entry['result'],
            entry['confidence'],
            entry['status'],
            entry['timestamp']
        ])

    csv_data = output.getvalue()
    output.close()

    filename = f"pneumonia_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        csv_data,
        mimetype='text/csv',
        headers={
            'Content-Disposition': f'attachment; filename={filename}'
        }
    )

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🏥 PNEUMONIA DETECTION WEB APP")
    print("="*50)
    print(f"📍 URL: http://localhost:5000")
    print(f"🤖 Model: {MODEL_PATH}")
    print("="*50 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
