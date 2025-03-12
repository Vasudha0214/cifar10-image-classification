# Part 2: Model Training & Evaluation
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2, ResNet50
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pickle
import time

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load preprocessed data
def load_preprocessed_data():
    try:
        x_train = np.load('x_train_norm.npy')
        y_train = np.load('y_train_final.npy')
        x_val = np.load('x_val_norm.npy')
        y_val = np.load('y_val.npy')
        x_test = np.load('x_test_norm.npy')
        y_test = np.load('y_test.npy')
        
        # Load class names
        class_names = []
        with open('class_names.txt', 'r') as f:
            for line in f:
                class_names.append(line.strip())
        
        print("Preprocessed data loaded successfully!")
        return x_train, y_train, x_val, y_val, x_test, y_test, class_names
    except:
        print("Error loading preprocessed data. Run the preprocessing script first.")
        return None, None, None, None, None, None, None

x_train, y_train, x_val, y_val, x_test, y_test, class_names = load_preprocessed_data()

# Create a custom CNN model
def create_custom_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
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
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create a MobileNetV2 model with transfer learning
def create_mobilenetv2_model(input_shape=(32, 32, 3), num_classes=10):
    # Load MobileNetV2 with pre-trained weights (exclude top layers)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create new model on top
    inputs = Input(shape=input_shape)
    # Apply normalization expected by MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Fix: Add the missing GlobalAveragePooling2D import
from tensorflow.keras.layers import GlobalAveragePooling2D

# Set up callbacks for training
def setup_callbacks(model_name):
    # Create model directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Learning rate reduction when validation loss plateaus
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Save the best model
    checkpoint = ModelCheckpoint(
        f'models/{model_name}_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    return [early_stopping, reduce_lr, checkpoint]

# Train a model with data augmentation
def train_model_with_augmentation(model, model_name, x_train, y_train, x_val, y_val, batch_size=64, epochs=50):
    print(f"\nTraining {model_name} model...")
    
    # Set up data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    
    # Initialize the data generator
    datagen.fit(x_train)
    
    # Set up callbacks
    callbacks = setup_callbacks(model_name)
    
    # Start timing
    start_time = time.time()
    
    # Train the model with data augmentation
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(x_train) // batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # End timing
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    return model, history

# Train a custom CNN model
def train_custom_cnn(x_train, y_train, x_val, y_val):
    # Create model
    model = create_custom_cnn_model()
    
    # Print model summary
    print("Custom CNN Model Summary:")
    model.summary()
    
    # Train model
    model, history = train_model_with_augmentation(
        model, 
        'custom_cnn', 
        x_train, 
        y_train, 
        x_val, 
        y_val
    )
    
    return model, history

# Plot training history
def plot_training_history(history, model_name):
    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_name} - Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} - Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_training_history.png')
    plt.close()

# Evaluate the model
def evaluate_model(model, x_test, y_test, class_names, model_name):
    print(f"\nEvaluating {model_name} model...")
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f'Test accuracy: {test_acc:.4f}')
    print(f'Test loss: {test_loss:.4f}')
    
    # Predict probabilities for test set
    y_pred_prob = model.predict(x_test)
    
    # Convert probabilities to class predictions
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_confusion_matrix.png')
    plt.close()
    
    # Generate classification report
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names, 
        output_dict=True
    )
    
    # Convert classification report to DataFrame for easier visualization
    report_df = pd.DataFrame(report).transpose()
    
    # Save classification report
    report_df.to_csv(f'plots/{model_name}_classification_report.csv')
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Plot precision, recall, and F1-score for each class
    plt.figure(figsize=(12, 8))
    metrics_df = report_df.iloc[:-3]  # Exclude the avg rows
    
    # Plot precision
    plt.subplot(3, 1, 1)
    sns.barplot(x=metrics_df.index, y=metrics_df['precision'])
    plt.title(f'{model_name} - Precision by Class')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Plot recall
    plt.subplot(3, 1, 2)
    sns.barplot(x=metrics_df.index, y=metrics_df['recall'])
    plt.title(f'{model_name} - Recall by Class')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Plot F1-score
    plt.subplot(3, 1, 3)
    sns.barplot(x=metrics_df.index, y=metrics_df['f1-score'])
    plt.title(f'{model_name} - F1-Score by Class')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_metrics_by_class.png')
    plt.close()
    
    # Create a dictionary of evaluation results
    evaluation_results = {
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return evaluation_results

# Show misclassified examples
def show_misclassified_examples(model, x_test, y_test, class_names, model_name, num_examples=20):
    # Predict test data
    y_pred_prob = model.predict(x_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Find misclassified examples
    misclassified_indices = np.where(y_pred != y_true)[0]
    
    # If there are fewer misclassified examples than requested, adjust
    num_examples = min(num_examples, len(misclassified_indices))
    
    # Randomly select examples to show
    selected_indices = np.random.choice(misclassified_indices, num_examples, replace=False)
    
    # Plot misclassified examples
    rows = int(np.ceil(num_examples / 5))
    plt.figure(figsize=(15, rows * 3))
    
    for i, idx in enumerate(selected_indices):
        plt.subplot(rows, 5, i+1)
        plt.imshow(x_test[idx])
        plt.title(f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_misclassified.png')
    plt.close()

# Save the trained model
def save_model(model, model_name):
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save the entire model (architecture + weights + optimizer state)
    model.save(f'models/{model_name}_full_model')
    
    # Save as TensorFlow SavedModel format
    model.save(f'models/{model_name}_saved_model')
    
    # Save model in ONNX format
    try:
        import tf2onnx
        import onnx
        
        onnx_model, _ = tf2onnx.convert.from_keras(model)
        onnx.save(onnx_model, f'models/{model_name}.onnx')
        print(f"Model saved in ONNX format: models/{model_name}.onnx")
    except:
        print("Could not save in ONNX format. Please install tf2onnx and onnx.")
    
    # Also save model architecture and weights separately
    model_json = model.to_json()
    with open(f'models/{model_name}_architecture.json', 'w') as json_file:
        json_file.write(model_json)
    
    # Save weights
    model.save_weights(f'models/{model_name}_weights.h5')
    
    print(f"Model saved to models/{model_name}")

def run_model_training_pipeline():
    # Load preprocessed data
    x_train, y_train, x_val, y_val, x_test, y_test, class_names = load_preprocessed_data()
    
    if x_train is None:
        return
    
    # Train the custom CNN model
    custom_model, custom_history = train_custom_cnn(x_train, y_train, x_val, y_val)
    
    # Plot training history
    plot_training_history(custom_history, 'custom_cnn')
    
    # Evaluate the model
    evaluation_results = evaluate_model(custom_model, x_test, y_test, class_names, 'custom_cnn')
    
    # Show misclassified examples
    show_misclassified_examples(custom_model, x_test, y_test, class_names, 'custom_cnn')
    
    # Save the model
    save_model(custom_model, 'custom_cnn')
    
    print("\nModel training and evaluation complete!")
    return custom_model, evaluation_results

# Run the training pipeline if this file is executed directly
if __name__ == "__main__":
    model, evaluation_results = run_model_training_pipeline()