import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
def load_data(path):
    """
    Membaca data dari path yang diberikan.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan di path: {path}")
    
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

def clean_and_encode_data(df):
    """
    Melakukan cleaning (TotalCharges) dan encoding (One-Hot/Label).
    """
    print("Starting data cleaning and encoding...")
    
    # Drop kolom tidak digunakan
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # Mengubah TotalCharges dari Object menjadi Numeric & Handle Missing Values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    
    # Encoding Target 'Churn' (Yes = 1, No = 0)
    if 'Churn' in df.columns:
        le = LabelEncoder()
        df['Churn'] = le.fit_transform(df['Churn'])
        print("Target 'Churn' encoded.")

    # One-Hot Encoding untuk fitur kategorial lainnya
    cat_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    print(f"Data processed. New Shape: {df.shape}")
    return df

def split_and_scale(df, target_col='Churn', test_size=0.2):
    """
    Membagi data train/test dan melakukan scaling.
    """    
    # Split data menjadi Fitur (X) dan Target (y)
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Split Data menjadi 80% Training, 20% Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Melakukan Standarisasi untuk fitur numerik
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    # Kembalikan ke DataFrame
    train_set = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
    test_set = pd.concat([X_test_scaled, y_test.reset_index(drop=True)], axis=1)
    
    return train_set, test_set

def main():
    # Setup Path (Fleksibel untuk Lokal maupun GitHub Actions)
    # Asumsi struktur: root/preprocessing/automate.py dan root/data_raw/data.csv
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)                  
    
    # Lokasi Input & Output
    input_path = os.path.join(root_dir, 'telco_customer_churn_raw', 'telco_customer_churn_raw.csv')
    output_path = os.path.join(base_dir, "telco_customer_churn_preprocessing")
    
    # Buat folder output jika belum ada
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # Eksekusi Pipeline
        df = load_data(input_path)
        df_clean = clean_and_encode_data(df)
        train_data, test_data = split_and_scale(df_clean)
        
        # Simpan Hasil
        train_data.to_csv(os.path.join(output_path, 'train_clean.csv'), index=False)
        test_data.to_csv(os.path.join(output_path, 'test_clean.csv'), index=False)
        
        print(f"SUCCESS! Data saved to: {output_path}")
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()