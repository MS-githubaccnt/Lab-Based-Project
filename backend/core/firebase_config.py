import firebase_admin
from firebase_admin import credentials
import os

def initialize_firebase():
    """
    Initializes the Firebase Admin SDK. 
    Call this once when the FastAPI server starts.
    """
    if not firebase_admin._apps:
        try:
            cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json")
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            print("🔥 Firebase Admin SDK initialized successfully.")
        except Exception as e:
            print(f"❌ Failed to initialize Firebase Admin SDK: {e}")
            print("Please make sure 'serviceAccountKey.json' is in the correct path.")
            raise e