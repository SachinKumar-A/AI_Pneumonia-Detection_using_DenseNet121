"""
Train Pneumonia Detection Model with GPU Support
This script trains a DenseNet121 model for pneumonia detection with automatic GPU acceleration
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np
from sklearn.model_selection import train_test_split
import time
from datetime import datetime

# Configuration
BATCH_SIZE = 16  # Reduced from 32 to fit RTX 3050 (6GB VRAM)
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
SEED = 42
EPOCHS = 30  # Can be changed for faster/slower training
LEARNING_RATE = 1e-3
TRAIN_SPLIT = 0.8

# Data directories
DATA_DIR = './chest_xray'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Model save path
MODEL_PATH = 'model2result.keras'

# Set random seeds for reproducibility
tf.random.set_seed(SEED)
np.random.seed(SEED)

def setup_gpu():
    """Setup GPU configuration for training"""
    print("\n" + "="*60)
    print("GPU CONFIGURATION")
    print("="*60)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to prevent TensorFlow from allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"[OK] GPU Available: {len(gpus)} GPU(s) detected")
            for i, gpu in enumerate(gpus):
                print(f"   - GPU {i}: {gpu.name}")
            
            # Print GPU details
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            print(f"   - Compute Capability: {gpu_details.get('compute_capability', 'N/A')}")
            
            return True
        except RuntimeError as e:
            print(f"[WARN] GPU setup error: {e}")
            return False
    else:
        print("[WARN] No GPU detected - Training will use CPU (slower)")
        print("   To enable GPU: Install CUDA Toolkit and cuDNN")
        print("   Visit: https://www.tensorflow.org/install/pip#windows-native")
        return False

def load_dataset():
    """Load and prepare the dataset"""
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    
    # Get class names (only directories)
    class_names = sorted([d for d in os.listdir(TRAIN_DIR) 
                         if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    num_classes = len(class_names)
    
    print(f"Classes: {class_names}")
    print(f"Number of classes: {num_classes}")
    
    # Load training images
    train_images = []
    train_labels = []
    
    for class_name in class_names:
        class_dir = os.path.join(TRAIN_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                train_images.append(img_path)
                train_labels.append(class_to_idx[class_name])
    
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    
    print(f"\nTotal training images: {len(train_images)}")
    
    # Split into train and validation
    new_train_paths, new_val_paths, new_train_labels, new_val_labels = train_test_split(
        train_images,
        train_labels,
        test_size=(1 - TRAIN_SPLIT),
        stratify=train_labels,
        random_state=SEED
    )
    
    print(f"Training set: {len(new_train_paths)} images ({TRAIN_SPLIT*100:.0f}%)")
    print(f"Validation set: {len(new_val_paths)} images ({(1-TRAIN_SPLIT)*100:.0f}%)")
    
    return new_train_paths, new_val_paths, new_train_labels, new_val_labels, class_names, num_classes

def create_dataset(image_paths, labels, augment=False, shuffle=True):
    """Create TF dataset from paths and labels with GPU optimization"""
    
    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        return img, label
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    # Use AUTOTUNE for optimal performance
    AUTOTUNE = tf.data.AUTOTUNE
    
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, seed=SEED)
    
    dataset = dataset.batch(BATCH_SIZE)
    
    # Apply data augmentation
    if augment:
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomTranslation(0.1, 0.1),
        ])
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )
    
    # Apply preprocessing
    dataset = dataset.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=AUTOTUNE
    )
    
    # Prefetch for better performance
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset

def build_model(num_classes):
    """Build DenseNet121 model for pneumonia detection"""
    print("\n" + "="*60)
    print("BUILDING MODEL")
    print("="*60)
    
    base_model = DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=INPUT_SHAPE,
        pooling='avg'
    )
    
    inputs = base_model.input
    x = base_model.output
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="DenseNet121_Pneumonia")
    
    print(f"Model: {model.name}")
    print(f"Total parameters: {model.count_params():,}")
    
    return model

def train_model(model, train_ds, val_ds):
    """Train the model with callbacks"""
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Model will be saved to: {MODEL_PATH}")
    print("="*60)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=f'logs/fit/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        )
    ]
    
    # Train
    start_time = time.time()
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Total training time: {training_time/60:.2f} minutes")
    print(f"Average time per epoch: {training_time/EPOCHS:.2f} seconds")
    print(f"Best model saved to: {MODEL_PATH}")
    
    # Print best results
    best_epoch = np.argmax(history.history['val_accuracy'])
    print(f"\nBest results at epoch {best_epoch + 1}:")
    print(f"  - Training Accuracy: {history.history['accuracy'][best_epoch]:.4f}")
    print(f"  - Validation Accuracy: {history.history['val_accuracy'][best_epoch]:.4f}")
    print(f"  - Training Loss: {history.history['loss'][best_epoch]:.4f}")
    print(f"  - Validation Loss: {history.history['val_loss'][best_epoch]:.4f}")
    
    return history

def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("PNEUMONIA DETECTION MODEL TRAINING")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup GPU
    has_gpu = setup_gpu()
    
    # Load dataset
    train_paths, val_paths, train_labels, val_labels, class_names, num_classes = load_dataset()
    
    # Create datasets
    print("\n" + "="*60)
    print("PREPARING DATA PIPELINES")
    print("="*60)
    train_ds = create_dataset(train_paths, train_labels, augment=True, shuffle=True)
    val_ds = create_dataset(val_paths, val_labels, augment=False, shuffle=False)
    print("[OK] Data pipelines ready")
    
    # Build model
    model = build_model(num_classes)
    
    # Train model
    history = train_model(model, train_ds, val_ds)
    
    print("\n" + "="*60)
    print("[SUCCESS] TRAINING COMPLETE!")
    print("="*60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nModel saved at: {MODEL_PATH}")
    print(f"Model size: {os.path.getsize(MODEL_PATH) / (1024**2):.2f} MB")
    print("\nYou can now use this model with the Streamlit app!")
    print("Run: py -3 -m streamlit run streamlit_app.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
