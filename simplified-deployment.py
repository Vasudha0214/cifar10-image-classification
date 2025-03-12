# app.py - Simplified FastAPI deployment script
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
import secrets
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("image-classifier-api")

# Initialize FastAPI app
app = FastAPI(
    title="CIFAR-10 Image Classification API",
    description="API for classifying images using a trained CNN model on CIFAR-10 dataset",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup basic authentication
security = HTTPBasic()

# Authentication credentials
API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "password")

# Create model architecture (to be used if loading weights directly)
def create_model_architecture():
    """Create the same model architecture to load weights"""
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        
        # Second convolutional block
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        
        # Third convolutional block
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),
        
        # Flatten and dense layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=8e-6),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Model and class names global variables
model = None
class_names = []

# Function to verify credentials
def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, API_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, API_PASSWORD)
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# Load model and class names
def load_model_and_classes():
    global model, class_names
    
    # Try multiple approaches to load the model
    try:
        # Approach 1: Try loading from 'deployment_model' directory (if available)
        if os.path.exists('models/deployment_model'):
            logger.info("Loading model from models/deployment_model...")
            model = load_model('models/deployment_model')
            logger.info("Model loaded successfully from deployment_model directory")
        
        # Approach 2: Try loading the Keras or H5 file directly
        elif os.path.exists('models/custom_cnn_best.h5') or os.path.exists('models/custom_cnn_best.keras'):
            try:
                if os.path.exists('models/custom_cnn_best.keras'):
                    logger.info("Loading model from models/custom_cnn_best.keras...")
                    model = load_model('models/custom_cnn_best.keras')
                else:
                    logger.info("Loading model from models/custom_cnn_best.h5...")
                    model = load_model('models/custom_cnn_best.h5')
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load model file directly: {e}")
                
                # Approach 3: Create a new model and load weights
                logger.info("Creating a new model and loading weights...")
                model = create_model_architecture()
                if os.path.exists('models/custom_cnn_best.keras'):
                    model.load_weights('models/custom_cnn_best.keras')
                else:
                    model.load_weights('models/custom_cnn_best.h5')
                logger.info("Model weights loaded successfully")
        else:
            logger.error("No model files found")
            raise RuntimeError("No model files found in the models directory")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

    # Load class names
    try:
        logger.info("Loading class names...")
        # CIFAR-10 class names (hardcoded as fallback)
        default_class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        # Try to load from file first
        if os.path.exists('class_names.txt'):
            with open('class_names.txt', 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
            logger.info(f"Loaded {len(class_names)} class names from file")
        else:
            # Use default CIFAR-10 class names
            class_names = default_class_names
            logger.info("Using default CIFAR-10 class names")
    except Exception as e:
        logger.warning(f"Error loading class names from file: {e}")
        # Use default CIFAR-10 class names
        class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        logger.info("Using default CIFAR-10 class names as fallback")

# Load on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the API server...")
    load_model_and_classes()

# Preprocess image function
def preprocess_image(image: Image.Image, target_size=(32, 32)):
    """
    Preprocess an image for the model (resize to 32x32 and normalize)
    """
    # Resize image
    image = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(image)
    
    # Handle grayscale images (convert to RGB)
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    
    # Handle RGBA images (convert to RGB)
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    # Normalize pixel values to [0, 1]
    img_array = img_array.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running
    """
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "class_names": class_names
    }

# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "CIFAR-10 Image Classification API",
        "version": "1.0.0",
        "model_info": "Custom CNN trained on CIFAR-10 dataset",
        "class_names": class_names,
        "endpoints": {
            "/health": "Health check endpoint",
            "/predict": "Image classification endpoint (requires authentication)",
            "/docs": "API documentation"
        }
    }

# Predict endpoint
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    username: str = Depends(verify_credentials)
):
    """
    Predict the class of an uploaded image
    """
    start_time = time.time()
    
    try:
        logger.info(f"Received prediction request from user: {username}")
        
        # Read and preprocess the image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        processed_image = preprocess_image(image)
        
        # Make prediction
        if model is None:
            logger.error("Model not loaded")
            raise HTTPException(
                status_code=500, 
                detail="Model not loaded. Please try again later."
            )
        
        predictions = model.predict(processed_image)[0]
        
        # Get the predicted class and confidence
        predicted_class_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_class_idx])
        predicted_class = class_names[predicted_class_idx]
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions)[-3:][::-1]
        top_predictions = {
            class_names[idx]: float(predictions[idx]) 
            for idx in top_indices
        }
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        logger.info(f"Prediction: {predicted_class}, Confidence: {confidence:.4f}, Time: {processing_time:.4f}s")
        
        # Return prediction results
        return {
            "class_name": predicted_class,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

# Run the application
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
