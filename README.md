# CIFAR-10 Image Classification Project

A complete machine learning project implementing CIFAR-10 image classification, from data preprocessing to model deployment.

## Project Structure

```
.
├── python_preprocessing.py     # Data preprocessing and EDA
├── model_train_eval.py        # Model training and evaluation
├── simplified-deployment.py    # FastAPI deployment
├── requirements.txt           # Project dependencies
├── Dockerfile                 # Container configuration
├── .dockerignore             # Docker build exclusions
└── models/                   # Model files
    └── custom_cnn_best.keras
```

## 1. Data Preprocessing (`python_preprocessing.py`)

### Features
- Dataset loading (CIFAR-10)
- Exploratory Data Analysis (EDA)
  - Class distribution visualization
  - Sample image display
  - Pixel value distribution analysis
- Data preprocessing
  - Normalization (0-1 range)
  - Train/validation split
  - One-hot encoding
- Data augmentation
  - Random rotation (±15°)
  - Width/height shifts (10%)
  - Horizontal flips
  - Zoom range (10%)
- TensorFlow dataset creation
- Preprocessing results saving

### Usage
```bash
python python_preprocessing.py
```

## 2. Model Training & Evaluation (`model_train_eval.py`)

### Model Architecture
```
- Conv2D(32) + BatchNorm + ReLU
- Conv2D(32) + BatchNorm + ReLU
- MaxPooling + Dropout(0.2)
- Conv2D(64) + BatchNorm + ReLU
- Conv2D(64) + BatchNorm + ReLU
- MaxPooling + Dropout(0.3)
- Conv2D(128) + BatchNorm + ReLU
- Conv2D(128) + BatchNorm + ReLU
- MaxPooling + Dropout(0.4)
- Dense(256) + BatchNorm + ReLU
- Dropout(0.5)
- Dense(10) + Softmax
```

### Training Features
- Custom CNN architecture
- Data augmentation during training
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Training history visualization
- Comprehensive evaluation
  - Confusion matrix
  - Classification report
  - Per-class metrics
  - Misclassified examples

### Usage
```bash
python model_train_eval.py
```

## 3. API Deployment (`simplified-deployment.py`)

### Features
- FastAPI-based REST API
- Model loading and inference
- Image preprocessing
- Authentication
- Error handling
- Request logging
- CORS support
- Health check endpoint
- Swagger documentation

### API Endpoints

1. **Root** (`GET /`)
   - Returns API information

2. **Health Check** (`GET /health`)
   - Checks API and model status

3. **Predict** (`POST /predict`)
   - Image classification endpoint
   - Requires authentication
   - Returns predictions with confidence scores

### Deployment Options

#### Using Docker
```bash
# Build Docker image
docker build -t cifar10-classifier .

# Run container
docker run -p 8000:8000 cifar10-classifier
```

#### Without Docker
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run API
uvicorn simplified-deployment:app --reload
```

### Testing the API
```bash
# Test prediction endpoint
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -u "admin:password" \
  -F "file=@your_image.jpg"
```

Example Response:
```json
{
  "class_name": "automobile",
  "confidence": 0.9999771118164062,
  "top_predictions": {
    "automobile": 0.9999771118164062,
    "truck": 2.290104384883307e-05,
    "ship": 3.5734930570185952e-09
  },
  "processing_time": 0.24216604232788086
}
```

## Dependencies

- Python 3.11
- TensorFlow 2.15.0
- FastAPI 0.109.2
- Uvicorn 0.27.1
- NumPy 1.24.3
- Pillow 10.2.0
- See `requirements.txt` for full list

## Model Performance

The model achieves competitive performance on CIFAR-10:
- Training with data augmentation
- Early stopping to prevent overfitting
- Batch normalization for stability
- Dropout for regularization

## Security Features

- Basic authentication
- CORS middleware
- Input validation
- Error handling
- Secure file handling

## Bonus Features

1. **Comprehensive Logging**
   - File and console handlers
   - Request/response logging
   - Error tracking
   - Performance monitoring

2. **Docker Support**
   - Multi-stage builds
   - Optimized image size
   - Environment configuration
   - Port mapping

## Future Improvements

1. Environment variable configuration
2. Model versioning
3. Rate limiting
4. Request validation
5. Caching layer
6. Load balancing
7. Model retraining pipeline
8. A/B testing support

## Running the Complete Pipeline

1. Preprocess the data:
```bash
python python_preprocessing.py
```

2. Train and evaluate the model:
```bash
python model_train_eval.py
```

3. Deploy the API:
```bash
uvicorn simplified-deployment:app --reload
```

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements. 