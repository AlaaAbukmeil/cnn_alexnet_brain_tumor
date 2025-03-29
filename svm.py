import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import time
import joblib
import pickle

dataset_path = 'brain-tumor-mri-dataset'
train_path = os.path.join(dataset_path, 'Training')
test_path = os.path.join(dataset_path, 'Testing')

SAVED_MODEL_PATH = 'brain_tumor_svm_model.pkl'
SAVED_FEATURES_PATH = 'extracted_features/'
os.makedirs(SAVED_FEATURES_PATH, exist_ok=True)

img_size = (150, 150) 
batch_size = 32
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

def extract_features(directory, feature_extractor, save_path=None):
    if save_path and os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            features_data = pickle.load(f)
        return features_data['features'], features_data['labels']
    
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    generator = datagen.flow_from_directory(
        directory,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    features = []
    labels = []
    sample_count = 0
    max_samples = len(generator.filenames)
    
    for i in range(int(np.ceil(max_samples / batch_size))):
        if sample_count >= max_samples:
            break
        
        batch = next(generator)
        batch_features = feature_extractor.predict(batch[0], verbose=0)
        batch_features = batch_features.reshape(batch_features.shape[0], -1)
        
        batch_labels = np.argmax(batch[1], axis=1)
        
        features.append(batch_features)
        labels.append(batch_labels)
        
        sample_count += len(batch_labels)
    
    features_array = np.vstack(features)
    labels_array = np.concatenate(labels)
    
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump({
                'features': features_array,
                'labels': labels_array,
                'class_names': generator.class_indices
            }, f)
    
    return features_array, labels_array

def create_feature_extractor():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*img_size, 3), pooling='avg')
    return Model(inputs=base_model.input, outputs=base_model.output)

def calculate_class_metrics(cm, class_names):
    n_classes = len(class_names)
    class_metrics = {}
    
    # Total number of examples
    total = np.sum(cm)
    
    for i, class_name in enumerate(class_names):
        # True positives: correctly predicted as class i
        tp = cm[i, i]
        
        # False negatives: incorrectly predicted as not class i (but were class i)
        fn = np.sum(cm[i, :]) - tp
        
        # False positives: incorrectly predicted as class i (but were not class i)
        fp = np.sum(cm[:, i]) - tp
        
        # True negatives: correctly predicted as not class i
        tn = total - (tp + fp + fn)
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Store metrics
        class_total = tp + fn  # Total examples of this class
        class_metrics[class_name] = {
            'accuracy': (accuracy, f"{tp + tn}/{total}"),
            'precision': (precision, f"{tp}/{tp + fp}"),
            'recall': (recall, f"{tp}/{class_total}"),
            'specificity': (specificity, f"{tn}/{tn + fp}"),
            'f1': f1,
        }
    
    return class_metrics

if __name__ == "__main__":
    start_time = time.time()
    
    if os.path.exists(SAVED_MODEL_PATH):
        print(f"Loading pre-trained SVM model from {SAVED_MODEL_PATH}")
        best_model = joblib.load(SAVED_MODEL_PATH)
        model_loaded = True
    else:
        model_loaded = False
    
    train_features_path = os.path.join(SAVED_FEATURES_PATH, 'train_features.pkl')
    test_features_path = os.path.join(SAVED_FEATURES_PATH, 'test_features.pkl')
    
    feature_extractor = create_feature_extractor()
    
    X_train, y_train = extract_features(train_path, feature_extractor, train_features_path)
    X_test, y_test = extract_features(test_path, feature_extractor, test_features_path)
    
    print(f"Feature processing completed in {time.time() - start_time:.2f} seconds")
    print(f"Training features shape: {X_train.shape}")
    print(f"Testing features shape: {X_test.shape}")
    
    if not model_loaded:
        svm_time_start = time.time()
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', probability=True, class_weight='balanced'))
        ])
        
        param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto', 0.01, 0.001]
        }
        
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=3, 
            scoring='accuracy',
            verbose=2,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        joblib.dump(best_model, SAVED_MODEL_PATH)
    
    y_pred = best_model.predict(X_test)
    test_accuracy = best_model.score(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - SVM')
    plt.savefig('svm_confusion_matrix.png')
    plt.show()

    metrics = calculate_class_metrics(cm, class_names)
    
    for class_name in class_names:
        print(f"{class_name.capitalize()}")
        
        class_metrics = metrics[class_name]
        
        acc, acc_ratio = class_metrics['accuracy']
        print(f"Accuracy: {acc*100:.2f}% ({acc_ratio})")
        
        prec, prec_ratio = class_metrics['precision']
        print(f"Precision: {prec*100:.2f}% ({prec_ratio})")
        
        rec, rec_ratio = class_metrics['recall']
        print(f"Recall/Sensitivity: {rec*100:.2f}% ({rec_ratio})")
        
        spec, spec_ratio = class_metrics['specificity']
        print(f"Specificity: {spec*100:.2f}% ({spec_ratio})")
        
        print(f"F1 Score: {class_metrics['f1']*100:.2f}%")
        print()
    