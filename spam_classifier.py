import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import joblib
from text_preprocessor import TextPreprocessor

class SpamClassifier:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.nb_model = None
        self.svm_model = None
        self.vectorizer = None
        self.X_test = None
        self.y_test = None
        
    def prepare_data(self, data):
        """Prepare data for training"""
        # Extract text and labels
        texts = data['text'].values
        labels = data['label'].values
        
        # Convert labels to binary (0 for ham, 1 for spam)
        y = np.array([1 if label == 'spam' else 0 for label in labels])
        
        # Preprocess and vectorize text
        X = self.preprocessor.fit_transform(texts)
        self.vectorizer = self.preprocessor.vectorizer
        
        return X, y
    
    def train_naive_bayes(self, X, y):
        """Train Naive Bayes classifier"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Store test data for later use
        self.X_test = X_test
        self.y_test = y_test
        
        # Train Naive Bayes
        self.nb_model = MultinomialNB(alpha=1.0)
        self.nb_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.nb_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'classification_report': classification_report(y_test, y_pred)
        }
    
    def train_svm(self, X, y):
        """Train SVM classifier"""
        # Use the same test split as Naive Bayes
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train SVM with probability estimates
        self.svm_model = SVC(kernel='linear', probability=True, C=1.0, random_state=42)
        self.svm_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.svm_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'classification_report': classification_report(y_test, y_pred)
        }
    
    def predict_naive_bayes(self, text):
        """Predict using Naive Bayes model"""
        if self.nb_model is None:
            raise ValueError("Naive Bayes model not trained yet")
        
        # Preprocess text
        processed_text = self.preprocessor.transform([text])
        
        # Get prediction and probability
        prediction = self.nb_model.predict(processed_text)[0]
        probabilities = self.nb_model.predict_proba(processed_text)[0]
        
        # Get confidence (probability of predicted class)
        confidence = probabilities[prediction]
        
        # Convert prediction back to label
        label = 'spam' if prediction == 1 else 'ham'
        
        return label, confidence
    
    def predict_svm(self, text):
        """Predict using SVM model"""
        if self.svm_model is None:
            raise ValueError("SVM model not trained yet")
        
        # Preprocess text
        processed_text = self.preprocessor.transform([text])
        
        # Get prediction and probability
        prediction = self.svm_model.predict(processed_text)[0]
        probabilities = self.svm_model.predict_proba(processed_text)[0]
        
        # Get confidence (probability of predicted class)
        confidence = probabilities[prediction]
        
        # Convert prediction back to label
        label = 'spam' if prediction == 1 else 'ham'
        
        return label, confidence
    
    def save_models(self, nb_path='naive_bayes_model.pkl', svm_path='svm_model.pkl', vectorizer_path='vectorizer.pkl'):
        """Save trained models"""
        if self.nb_model:
            joblib.dump(self.nb_model, nb_path)
        if self.svm_model:
            joblib.dump(self.svm_model, svm_path)
        if self.vectorizer:
            joblib.dump(self.vectorizer, vectorizer_path)
    
    def load_models(self, nb_path='naive_bayes_model.pkl', svm_path='svm_model.pkl', vectorizer_path='vectorizer.pkl'):
        """Load trained models"""
        try:
            self.nb_model = joblib.load(nb_path)
            self.svm_model = joblib.load(svm_path)
            self.vectorizer = joblib.load(vectorizer_path)
            return True
        except FileNotFoundError:
            return False
