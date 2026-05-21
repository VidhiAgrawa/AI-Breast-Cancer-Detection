import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

def generate_large_dataset():
    print("Downloading the official UCI Breast Cancer Dataset... 🌍")
    
    # 1. Load the built-in dataset
    official_data = load_breast_cancer()
    
    # 2. Create a Pandas table with the data
    medical_dataframe = pd.DataFrame(official_data.data, columns=official_data.feature_names)
    
    # 3. By default in scikit-learn, 0=Malignant and 1=Benign. 
    # We will flip this so 1=Malignant and 0=Benign to match our GUI! 🔄
    medical_dataframe['diagnosis'] = 1 - official_data.target
    
    # 4. Extract ONLY the 5 specific features we built our dashboard to read! 🩺
    selected_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
    final_dataset = medical_dataframe[selected_features].copy()
    
    # 5. Rename the columns to match our previous format
    final_dataset.columns = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
    
    # 6. Add the diagnosis column to the end
    final_dataset['diagnosis'] = medical_dataframe['diagnosis']
    
    # 7. Add a fake 'id' column at the beginning so our training script doesn't break! 🆔
    final_dataset.insert(0, 'id', np.arange(100000, 100000 + len(final_dataset)))
    
    # 8. Save the massive new file! 💾
    final_dataset.to_csv('breast_cancer_data.csv', index=False)
    print(f"Success! Overwritten breast_cancer_data.csv with {len(final_dataset)} patient records! 🎉")

if __name__ == "__main__":
    generate_large_dataset()