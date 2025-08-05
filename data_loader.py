import pandas as pd
import numpy as np
import urllib.request
import zipfile
import os
import io

class DataLoader:
    def __init__(self):
        self.data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
        self.data_file = "SMSSpamCollection"
        
    def download_data(self):
        """Download SMS Spam Collection dataset"""
        try:
            print("Downloading SMS Spam Collection dataset...")
            
            # Download the zip file
            urllib.request.urlretrieve(self.data_url, "smsspamcollection.zip")
            
            # Extract the dataset
            with zipfile.ZipFile("smsspamcollection.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            
            # Clean up
            os.remove("smsspamcollection.zip")
            
            return True
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            return False
    
    def load_data(self):
        """Load and preprocess the spam dataset"""
        try:
            # Try to load existing file first
            if not os.path.exists(self.data_file):
                # Download if not exists
                if not self.download_data():
                    # If download fails, create a sample dataset
                    return self.create_sample_dataset()
            
            # Read the SMS Spam Collection data
            data = pd.read_csv(self.data_file, sep='\t', header=None, 
                             names=['label', 'text'], encoding='latin-1')
            
            # Clean the data
            data = data.dropna()
            data = data.drop_duplicates()
            
            # Ensure labels are lowercase
            data['label'] = data['label'].str.lower()
            
            print(f"Loaded {len(data)} samples")
            print(f"Spam: {len(data[data['label'] == 'spam'])}")
            print(f"Ham: {len(data[data['label'] == 'ham'])}")
            
            return data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Return sample dataset as fallback
            return self.create_sample_dataset()
    
    def create_sample_dataset(self):
        """Create a sample dataset for demonstration"""
        print("Creating sample dataset...")
        
        # Sample spam messages
        spam_messages = [
            "FREE! Win a $1000 gift card! Click here now! Limited time offer!",
            "Congratulations! You've won $10,000! Call now to claim your prize!",
            "URGENT! Your account will be closed. Click link to verify now!",
            "Get rich quick! Make $5000 per week working from home!",
            "Free ringtones! Text STOP to opt out. Standard rates apply.",
            "You have been selected for a special offer! Call now!",
            "Claim your free iPhone! Limited time only! Act fast!",
            "WINNER! You've won a luxury vacation! Call to claim!",
            "Free money! No strings attached! Click here now!",
            "Exclusive deal just for you! Save 90% on everything!",
            "Your loan has been approved! No credit check required!",
            "Free prescription drugs! No doctor required!",
            "Make money fast! Work from home opportunity!",
            "You've inherited $1 million! Contact us immediately!",
            "Free casino bonus! Play now and win big!",
            "Diet pills that really work! Lose 30 pounds in 30 days!",
            "Credit repair guaranteed! Bad credit OK!",
            "Free laptop! Just pay shipping and handling!",
            "You're pre-approved! Apply for credit now!",
            "Free insurance quote! Save hundreds on car insurance!"
        ]
        
        # Sample ham (legitimate) messages
        ham_messages = [
            "Hi John, can we schedule a meeting for tomorrow to discuss the project?",
            "Thank you for your purchase. Your order has been shipped.",
            "Reminder: Your dentist appointment is tomorrow at 2 PM.",
            "The meeting has been moved to conference room B.",
            "Happy birthday! Hope you have a wonderful day!",
            "Your library books are due next week.",
            "Don't forget to pick up milk on your way home.",
            "The weather forecast shows rain for this weekend.",
            "Your subscription renewal is due next month.",
            "Please review the attached document and provide feedback.",
            "The restaurant reservation is confirmed for 7 PM tonight.",
            "Your flight has been delayed by 30 minutes.",
            "Thanks for helping me move last weekend!",
            "The conference call will start in 10 minutes.",
            "Your package has been delivered to your front door.",
            "The gym will be closed for maintenance on Sunday.",
            "Your prescription is ready for pickup at the pharmacy.",
            "The project deadline has been extended to next Friday.",
            "Your car service appointment is scheduled for Monday.",
            "The school holiday schedule has been updated on the website."
        ]
        
        # Create DataFrame
        labels = ['spam'] * len(spam_messages) + ['ham'] * len(ham_messages)
        texts = spam_messages + ham_messages
        
        data = pd.DataFrame({
            'label': labels,
            'text': texts
        })
        
        # Shuffle the data
        data = data.sample(frac=1).reset_index(drop=True)
        
        print(f"Created sample dataset with {len(data)} samples")
        print(f"Spam: {len(data[data['label'] == 'spam'])}")
        print(f"Ham: {len(data[data['label'] == 'ham'])}")
        
        return data
