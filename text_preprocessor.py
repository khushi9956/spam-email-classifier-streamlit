import re
import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

class TextPreprocessor:
    def __init__(self):
        self.vectorizer = None
        self.stemmer = PorterStemmer()
        self.download_nltk_data()
        
        # Load stop words
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            # Fallback stop words if NLTK data is not available
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
                "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
                'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
                'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
                'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 
                'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 
                'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 
                'before', 'after', 'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 
                'over', 'under', 'again', 'further', 'then', 'once'
            }
    
    def download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            print("Warning: Could not download NLTK data. Using fallback methods.")
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_stem(self, text):
        """Tokenize and stem text"""
        try:
            # Tokenize using NLTK
            tokens = word_tokenize(text)
        except:
            # Fallback tokenization
            tokens = text.split()
        
        # Remove stop words and apply stemming
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                try:
                    stemmed = self.stemmer.stem(token)
                    processed_tokens.append(stemmed)
                except:
                    processed_tokens.append(token)
        
        return ' '.join(processed_tokens)
    
    def preprocess_text(self, text):
        """Complete text preprocessing pipeline"""
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize and stem
        processed_text = self.tokenize_and_stem(cleaned_text)
        
        return processed_text
    
    def fit_transform(self, texts):
        """Fit vectorizer and transform texts"""
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,          # Limit vocabulary size
            min_df=2,                   # Ignore terms that appear in less than 2 documents
            max_df=0.95,                # Ignore terms that appear in more than 95% of documents
            ngram_range=(1, 2),         # Use unigrams and bigrams
            stop_words='english'        # Additional stop word removal
        )
        
        # Fit and transform
        X = self.vectorizer.fit_transform(processed_texts)
        
        return X.toarray()
    
    def transform(self, texts):
        """Transform new texts using fitted vectorizer"""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Transform using fitted vectorizer
        X = self.vectorizer.transform(processed_texts)
        
        return X.toarray()
    
    def get_feature_names(self):
        """Get feature names from vectorizer"""
        if self.vectorizer is None:
            return []
        
        try:
            return self.vectorizer.get_feature_names_out()
        except:
            # Fallback for older scikit-learn versions
            return self.vectorizer.get_feature_names()
    
    def get_vocabulary_size(self):
        """Get vocabulary size"""
        if self.vectorizer is None:
            return 0
        
        return len(self.vectorizer.vocabulary_)
