import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_cancer_detection_model():
    print("Loading dataset... 📂")
    cancer_dataset = pd.read_csv('breast_cancer_data.csv') 
    
    # 2. FIXED: Drop BOTH 'diagnosis' and 'id' so the AI only learns from the 5 medical stats! 🩺
    features = cancer_dataset.drop(['diagnosis', 'id'], axis=1)
    target_labels = cancer_dataset['diagnosis']
    
    x_train, x_test, y_train, y_test = train_test_split(features, target_labels, test_size=0.2, random_state=42)
    
    print("Training the AI model... 🧠")
    random_forest_model = RandomForestClassifier(n_estimators=100)
    random_forest_model.fit(x_train, y_train)
    
    predictions = random_forest_model.predict(x_test)
    model_accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {model_accuracy * 100:.2f}% ✅")
    
    joblib.dump(random_forest_model, 'ai_medical_model.pkl')
    print("Model saved successfully as ai_medical_model.pkl 💾")

if __name__ == "__main__":
    train_cancer_detection_model()