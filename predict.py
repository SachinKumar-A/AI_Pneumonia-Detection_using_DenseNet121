"""
Simple pneumonia detection script - Run model inference directly
"""

import os
import sys
from PIL import Image
import numpy as np

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Try to import TensorFlow with fallback
try:
    import tensorflow as tf
    from tensorflow.keras.applications.densenet import preprocess_input
    print("[OK] TensorFlow loaded successfully")
except ImportError as e:
    print(f"[ERROR] TensorFlow not available: {e}")
    print("Install with: pip install tensorflow")
    sys.exit(1)

def load_model(model_path='best_densenet_pneumonia.keras'):
    """Load the trained pneumonia detection model"""
    
    # If model_path is relative, look in the script's directory
    if not os.path.isabs(model_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, model_path)
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"[OK] Model loaded: {model_path}")
        return model
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return None

def predict_pneumonia(image_path, model):
    """Predict if chest X-ray shows pneumonia"""
    
    try:
        # Load image
        img = Image.open(image_path)
        print(f"[OK] Image loaded: {image_path}")
        
        # Convert to array and make it writable
        img_array = np.array(img, dtype=np.float32).copy()
        
        # Handle different image formats
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        elif len(img_array.shape) == 3 and img_array.shape[2] == 1:  # Single channel
            img_array = np.repeat(img_array, 3, axis=2)
        
        # Normalize to 0-1 range if not already
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
        
        # Resize to 224x224 (DenseNet input size)
        # Convert to uint8 for tf.image.resize
        img_uint8 = (img_array * 255).astype(np.uint8) if img_array.max() <= 1.0 else img_array.astype(np.uint8)
        img_resized = tf.image.resize(img_uint8, (224, 224))
        # Convert back to float and normalize
        img_resized = tf.cast(img_resized, tf.float32) / 255.0
        img_resized = np.expand_dims(np.array(img_resized), axis=0)
        
        # Preprocess with DenseNet preprocessing
        img_preprocessed = preprocess_input(img_resized.copy())
        
        # Make prediction
        prediction = float(model.predict(img_preprocessed, verbose=0)[0][0])
        
        # Format result
        print("\n" + "="*50)
        if prediction > 0.5:
            confidence = prediction * 100
            print(f"[WARNING] PNEUMONIA DETECTED")
            print(f"Confidence: {confidence:.1f}%")
            result = "PNEUMONIA"
        else:
            confidence = (1 - prediction) * 100
            print(f"[OK] NORMAL (No pneumonia)")
            print(f"Confidence: {confidence:.1f}%")
            result = "NORMAL"
        print("="*50 + "\n")
        
        return result, prediction
    
    except Exception as e:
        print(f"[ERROR] Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Main function for batch prediction"""
    
    print("Chest X-Ray Pneumonia Detection")
    print("= " * 25)
    
    # Load model
    model = load_model()
    if model is None:
        sys.exit(1)
    
    # Get image path from command line or prompt user
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Interactive mode
        print("\nUsage: python predict.py <image_path>")
        print("\nOr enter image path below:")
        image_path = input("Image path: ").strip()
    
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        sys.exit(1)
    
    # Make prediction
    result, confidence = predict_pneumonia(image_path, model)
    
    if result:
        print(f"Result: {result}")
        if result == "PNEUMONIA":
            print("[WARNING] Recommendation: Consult a radiologist for professional diagnosis")
        return 0
    
    return 1

if __name__ == '__main__':
    sys.exit(main())
