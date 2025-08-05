import streamlit as st

st.set_page_config(
    page_title="Spam Email Classifier",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'Get help': None,
        'Report a bug': None,
        'About': None
    }
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from spam_classifier import SpamClassifier
from data_loader import DataLoader
import time

# Page configuration
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="wide"
)

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
    st.session_state.models_trained = False
    st.session_state.training_results = None

def main():
    st.title("📧 Spam Email Classifier")
    st.markdown("**Machine Learning-based Email Classification using Naive Bayes and SVM**")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["🏠 Home", "📊 Train Models", "🔍 Test Classifier", "📈 Model Comparison"])
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Train Models":
        show_training_page()
    elif page == "🔍 Test Classifier":
        show_testing_page()
    elif page == "📈 Model Comparison":
        show_comparison_page()

def show_home_page():
    st.header("Welcome to the Spam Email Classifier")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Features")
        st.markdown("""
        - **Dual Algorithm Support**: Naive Bayes and SVM classifiers
        - **Text Preprocessing**: Advanced NLP techniques
        - **Model Evaluation**: Comprehensive performance metrics
        - **Interactive Testing**: Real-time email classification
        - **Performance Comparison**: Side-by-side model analysis
        """)
    
    with col2:
        st.subheader("🚀 Getting Started")
        st.markdown("""
        1. **Train Models**: Go to the training page to build classifiers
        2. **Test Emails**: Use the testing interface to classify emails
        3. **Compare Results**: Analyze model performance metrics
        4. **Optimize**: Fine-tune based on evaluation results
        """)
    
    st.subheader("📋 Dataset Information")
    if st.button("Load Dataset Info"):
        data_loader = DataLoader()
        try:
            data = data_loader.load_data()
            if data is not None:
                st.success("✅ Dataset loaded successfully!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Emails", len(data))
                with col2:
                    spam_count = len(data[data['label'] == 'spam'])
                    st.metric("Spam Emails", spam_count)
                with col3:
                    ham_count = len(data[data['label'] == 'ham'])
                    st.metric("Ham Emails", ham_count)
                
                # Show sample data
                st.subheader("📄 Sample Data")
                st.dataframe(data.head(10))
            else:
                st.error("❌ Failed to load dataset. Please check your internet connection.")
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")

def show_training_page():
    st.header("📊 Model Training")
    
    if st.button("🚀 Train Both Models", type="primary"):
        train_models()
    
    if st.session_state.models_trained and st.session_state.training_results:
        display_training_results()

def train_models():
    """Train both Naive Bayes and SVM models"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize classifier
        status_text.text("Initializing classifier...")
        progress_bar.progress(10)
        classifier = SpamClassifier()
        
        # Load data
        status_text.text("Loading dataset...")
        progress_bar.progress(20)
        data_loader = DataLoader()
        data = data_loader.load_data()
        
        if data is None:
            st.error("❌ Failed to load dataset. Cannot proceed with training.")
            return
        
        # Prepare data
        status_text.text("Preprocessing text data...")
        progress_bar.progress(40)
        X, y = classifier.prepare_data(data)
        
        # Train models
        status_text.text("Training Naive Bayes model...")
        progress_bar.progress(60)
        nb_results = classifier.train_naive_bayes(X, y)
        
        status_text.text("Training SVM model...")
        progress_bar.progress(80)
        svm_results = classifier.train_svm(X, y)
        
        # Store results
        st.session_state.classifier = classifier
        st.session_state.models_trained = True
        st.session_state.training_results = {
            'naive_bayes': nb_results,
            'svm': svm_results
        }
        
        progress_bar.progress(100)
        status_text.text("✅ Training completed successfully!")
        
        st.success("🎉 Both models trained successfully!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def display_training_results():
    """Display training results and metrics"""
    st.subheader("📈 Training Results")
    
    results = st.session_state.training_results
    
    # Metrics comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔢 Naive Bayes Results")
        nb_metrics = results['naive_bayes']
        st.metric("Accuracy", f"{nb_metrics['accuracy']:.4f}")
        st.metric("Precision", f"{nb_metrics['precision']:.4f}")
        st.metric("Recall", f"{nb_metrics['recall']:.4f}")
        st.metric("F1-Score", f"{nb_metrics['f1_score']:.4f}")
    
    with col2:
        st.subheader("⚙️ SVM Results")
        svm_metrics = results['svm']
        st.metric("Accuracy", f"{svm_metrics['accuracy']:.4f}")
        st.metric("Precision", f"{svm_metrics['precision']:.4f}")
        st.metric("Recall", f"{svm_metrics['recall']:.4f}")
        st.metric("F1-Score", f"{svm_metrics['f1_score']:.4f}")
    
    # Confusion matrices
    st.subheader("🔄 Confusion Matrices")
    col1, col2 = st.columns(2)
    
    with col1:
        st.text("Naive Bayes Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(nb_metrics['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Blues', xticklabels=['Ham', 'Spam'], 
                   yticklabels=['Ham', 'Spam'], ax=ax)
        ax.set_title('Naive Bayes')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        st.pyplot(fig)
    
    with col2:
        st.text("SVM Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(svm_metrics['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Reds', xticklabels=['Ham', 'Spam'], 
                   yticklabels=['Ham', 'Spam'], ax=ax)
        ax.set_title('SVM')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        st.pyplot(fig)

def show_testing_page():
    st.header("🔍 Test Email Classifier")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train the models first before testing.")
        if st.button("Go to Training Page"):
            st.session_state.page = "📊 Train Models"
            st.rerun()
        return
    
    # Input methods
    input_method = st.radio("Choose input method:", 
                           ["✍️ Type Email", "📝 Use Sample Emails"])
    
    if input_method == "✍️ Type Email":
        email_text = st.text_area("Enter email content:", 
                                height=200,
                                placeholder="Type or paste your email content here...")
        
        if st.button("🔍 Classify Email", type="primary") and email_text:
            classify_email(email_text)
    
    else:
        # Sample emails for testing
        sample_emails = {
            "Spam Example 1": "FREE! Win a $1000 gift card! Click here now! Limited time offer! Act fast!",
            "Spam Example 2": "Congratulations! You've won $10,000! Call now to claim your prize! Don't delay!",
            "Ham Example 1": "Hi John, can we schedule a meeting for tomorrow to discuss the project? Let me know what time works for you.",
            "Ham Example 2": "Thank you for your purchase. Your order #12345 has been shipped and will arrive within 3-5 business days.",
            "Ham Example 3": "Reminder: Your dentist appointment is scheduled for tomorrow at 2 PM. Please arrive 15 minutes early."
        }
        
        selected_sample = st.selectbox("Choose a sample email:", list(sample_emails.keys()))
        email_text = st.text_area("Email content:", 
                                value=sample_emails[selected_sample],
                                height=150)
        
        if st.button("🔍 Classify Email", type="primary"):
            classify_email(email_text)

def classify_email(email_text):
    """Classify a single email using both models"""
    try:
        classifier = st.session_state.classifier
        
        # Get predictions from both models
        nb_prediction, nb_confidence = classifier.predict_naive_bayes(email_text)
        svm_prediction, svm_confidence = classifier.predict_svm(email_text)
        
        # Display results
        st.subheader("📊 Classification Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔢 Naive Bayes")
            if nb_prediction == 'spam':
                st.error(f"🚨 SPAM (Confidence: {nb_confidence:.2%})")
            else:
                st.success(f"✅ HAM (Confidence: {nb_confidence:.2%})")
        
        with col2:
            st.subheader("⚙️ SVM")
            if svm_prediction == 'spam':
                st.error(f"🚨 SPAM (Confidence: {svm_confidence:.2%})")
            else:
                st.success(f"✅ HAM (Confidence: {svm_confidence:.2%})")
        
        # Consensus
        st.subheader("🤝 Consensus")
        if nb_prediction == svm_prediction:
            if nb_prediction == 'spam':
                st.error("🚨 Both models agree: This email is SPAM")
            else:
                st.success("✅ Both models agree: This email is HAM")
        else:
            st.warning("⚠️ Models disagree on classification")
        
        # Confidence visualization
        st.subheader("📊 Confidence Comparison")
        confidence_data = pd.DataFrame({
            'Model': ['Naive Bayes', 'SVM'],
            'Confidence': [nb_confidence, svm_confidence],
            'Prediction': [nb_prediction, svm_prediction]
        })
        
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['red' if pred == 'spam' else 'green' for pred in confidence_data['Prediction']]
        bars = ax.bar(confidence_data['Model'], confidence_data['Confidence'], color=colors, alpha=0.7)
        ax.set_ylabel('Confidence')
        ax.set_title('Model Confidence Comparison')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, conf in zip(bars, confidence_data['Confidence']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{conf:.2%}', ha='center', va='bottom')
        
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"❌ Classification failed: {str(e)}")

def show_comparison_page():
    st.header("📈 Model Comparison")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train the models first to see comparison.")
        return
    
    results = st.session_state.training_results
    
    # Performance metrics comparison
    st.subheader("📊 Performance Metrics Comparison")
    
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Naive Bayes': [
            results['naive_bayes']['accuracy'],
            results['naive_bayes']['precision'],
            results['naive_bayes']['recall'],
            results['naive_bayes']['f1_score']
        ],
        'SVM': [
            results['svm']['accuracy'],
            results['svm']['precision'],
            results['svm']['recall'],
            results['svm']['f1_score']
        ]
    })
    
    # Display metrics table
    st.dataframe(metrics_df.round(4))
    
    # Metrics comparison chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics_df['Metric']))
    width = 0.35
    
    ax.bar(x - width/2, metrics_df['Naive Bayes'], width, label='Naive Bayes', alpha=0.8)
    ax.bar(x + width/2, metrics_df['SVM'], width, label='SVM', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['Metric'])
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, metric in enumerate(metrics_df['Metric']):
        nb_val = metrics_df['Naive Bayes'][i]
        svm_val = metrics_df['SVM'][i]
        ax.text(i - width/2, nb_val + 0.01, f'{nb_val:.3f}', ha='center', va='bottom')
        ax.text(i + width/2, svm_val + 0.01, f'{svm_val:.3f}', ha='center', va='bottom')
    
    st.pyplot(fig)
    
    # Best model recommendation
    st.subheader("🏆 Model Recommendation")
    
    nb_avg = np.mean([results['naive_bayes']['accuracy'], results['naive_bayes']['precision'], 
                     results['naive_bayes']['recall'], results['naive_bayes']['f1_score']])
    svm_avg = np.mean([results['svm']['accuracy'], results['svm']['precision'], 
                      results['svm']['recall'], results['svm']['f1_score']])
    
    if nb_avg > svm_avg:
        st.success("🏆 **Naive Bayes** performs better overall")
        st.info(f"Average performance: {nb_avg:.4f}")
    elif svm_avg > nb_avg:
        st.success("🏆 **SVM** performs better overall") 
        st.info(f"Average performance: {svm_avg:.4f}")
    else:
        st.info("🤝 Both models perform equally well")
    
    # Detailed analysis
    st.subheader("🔍 Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Strengths")
        if results['naive_bayes']['precision'] > results['svm']['precision']:
            st.write("• Naive Bayes has higher precision")
        if results['naive_bayes']['recall'] > results['svm']['recall']:
            st.write("• Naive Bayes has higher recall")
        if results['svm']['precision'] > results['naive_bayes']['precision']:
            st.write("• SVM has higher precision")
        if results['svm']['recall'] > results['naive_bayes']['recall']:
            st.write("• SVM has higher recall")
    
    with col2:
        st.subheader("📝 Recommendations")
        if results['naive_bayes']['precision'] > 0.95:
            st.write("• Naive Bayes: Good for avoiding false positives")
        if results['svm']['recall'] > 0.95:
            st.write("• SVM: Good for catching all spam emails")
        
        st.write("• Consider ensemble methods for better performance")
        st.write("• Fine-tune hyperparameters for improvement")

if __name__ == "__main__":
    main()
