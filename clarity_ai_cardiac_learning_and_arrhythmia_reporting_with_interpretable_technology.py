# --- 0. SETUP ---
# You must run this cell first!
!pip install wfdb numpy pandas matplotlib seaborn scikit-learn lightgbm imbalanced-learn shap neurokit2 pywavelets

# --- 1. Imports ---
import wfdb  # For loading the MIT-BIH dataset
import numpy as np
import pandas as pd
import neurokit2 as nk  # For fiducial points and HRV
import pywt  # For wavelet features
import lightgbm as lgb
import shap  # For XAI
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

# --- 2. Configuration & Label Mapping ---

# Map symbols to 'Normal' (0) or 'Anomaly' (1)
binary_annotation_map = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal and bundle branch blocks
    'A': 1, 'a': 1, 'J': 1, 'S': 1,        # Supraventricular (APC)
    'V': 1, 'E': 1,                        # Ventricular (PVC)
    'F': 1,                                # Fusion
}

# Records to use from MIT-BIH (excluding paced beats)
RECORDS = [
    '100', '101', '103', '105', '106', '108', '109', '111', '112', '113',
    '114', '115', '116', '117', '118', '119', '121', '122', '123', '124',
    '200', '201', '202', '203', '205', '207', '208', '209', '210', '212',
    '213', '214', '215', '217', '219', '220', '221', '222', '223', '228',
    '230', '231', '232', '233', '234'
]

# --- 3. Feature Extraction (Corrected) ---

def extract_hybrid_features(segment, rr_intervals_samples, fs=360):
    """
    Extracts the hybrid features from a single ECG beat segment.
    (Corrected to handle HRV units)
    """
    features = {}
    
    # 1. Statistical Features (6 features)
    features['mean'] = np.mean(segment)
    features['std'] = np.std(segment)
    features['ptp'] = np.ptp(segment)
    features['min'] = np.min(segment)
    features['max'] = np.max(segment)
    features['mad'] = np.mean(np.abs(segment - features['mean']))
    
    # 2. Wavelet Features (10 features)
    try:
        coeffs = pywt.wavedec(segment, 'db4', level=4)
        for i, c in enumerate(coeffs):
            features[f'wavelet_energy_L{i}'] = np.sum(c**2)
            features[f'wavelet_entropy_L{i}'] = pywt.shannon_entropy(c)
    except Exception:
        for i in range(5):
            features[f'wavelet_energy_L{i}'] = 0
            features[f'wavelet_entropy_L{i}'] = 0

    # 3. Fiducial, HRV, and Non-Linear Features (8 features)
    try:
        # Fiducial Points (1 feature)
        features['qrs_duration'] = (len(segment) / 2) / fs * 1000 # Rough QRS estimate
        
        # --- FIX: Convert samples to milliseconds for NeuroKit ---
        rr_intervals_ms = (rr_intervals_samples / fs) * 1000
        
        if len(rr_intervals_ms) > 5:
            # Note: sampling_rate=1000 because data is now in ms
            hrv_time = nk.hrv_time(rr_intervals_ms, sampling_rate=1000)
            features['hrv_sdnn'] = hrv_time['HRV_SDNN'].iloc[0]
            features['hrv_rmssd'] = hrv_time['HRV_RMSSD'].iloc[0]
            
            hrv_freq = nk.hrv_frequency(rr_intervals_ms, sampling_rate=1000, vlf=False, hf=True, lf=True)
            features['hrv_lf_hf_ratio'] = hrv_freq['HRV_LFHF'].iloc[0]
            
            hrv_nl = nk.hrv_nonlinear(rr_intervals_ms, sampling_rate=1000)
            features['poincare_sd1'] = hrv_nl['HRV_SD1'].iloc[0]
            features['poincare_sd2'] = hrv_nl['HRV_SD2'].iloc[0]
        else:
            features['hrv_sdnn'] = 0
            features['hrv_rmssd'] = 0
            features['hrv_lf_hf_ratio'] = 0
            features['poincare_sd1'] = 0
            features['poincare_sd2'] = 0
    except Exception:
        # Fill with 0 if NeuroKit fails
        features['qrs_duration'] = 0
        features['hrv_sdnn'] = 0
        features['hrv_rmssd'] = 0
        features['hrv_lf_hf_ratio'] = 0
        features['poincare_sd1'] = 0
        features['poincare_sd2'] = 0
        
    # RR-intervals (2 features)
    if len(rr_intervals_samples) > 0:
        features['rr_interval_post'] = rr_intervals_samples[-1]
    else:
        features['rr_interval_post'] = 0
        
    if len(rr_intervals_samples) > 1:
        features['rr_interval_pre'] = rr_intervals_samples[-2]
    else:
        features['rr_interval_pre'] = 0

    # Fill NaNs with 0
    return pd.Series(features).fillna(0)

# --- 4. Data Loading (Corrected) ---

def load_and_segment_data(records, window_size=256, use_binary_map=True):
    """
    Loads data from wfdb, segments beats, and extracts features.
    (Corrected to provide proper HRV context)
    """
    all_features = []
    all_labels = []
    
    print(f"Loading and processing {len(records)} records...")
    
    for rec_name in records:
        print(f"Processing record: {rec_name}")
        
        # Download record and annotations
        record = wfdb.rdrecord(f'mitdb/{rec_name}', sampto=30*60*360) # 30 min
        annotation = wfdb.rdann(f'mitdb/{rec_name}', 'atr', sampto=30*60*360)
        
        signal = record.p_signal[:, 0] # Use lead 1
        r_peaks = annotation.sample
        symbols = annotation.symbol
        
        # Map labels
        if use_binary_map:
            labels = [binary_annotation_map.get(s) for s in symbols]
        else:
            labels = [multi_class_map.get(s) for s in symbols]
        
        # Segment beats and extract features
        half_window = window_size // 2
        
        for i in range(1, len(r_peaks) - 1): # Skip first and last beat
            label = labels[i]
            if label is None:
                continue # Skip unmapped annotations
                
            # Get segment
            start = r_peaks[i] - half_window
            end = r_peaks[i] + half_window
            if start < 0 or end > len(signal):
                continue
                
            segment = signal[start:end]
            
            # --- FIXED SECTION ---
            # Get HRV context (last 30 RR intervals)
            rr_context_indices = np.where(r_peaks < r_peaks[i])[0]
            # Need at least 10 beats to get a stable HRV
            if len(rr_context_indices) < 10: 
                continue
            
            # Get the indices of the last 30 R-peaks
            context_indices = rr_context_indices[-30:]
            
            # Calculate the RR intervals (in samples) from these peaks
            last_rrs = np.diff(r_peaks[context_indices]) 
            # ---------------------
            
            # Extract features (pass rr_intervals in samples)
            features = extract_hybrid_features(segment, last_rrs, fs=record.fs)
            all_features.append(features)
            all_labels.append(label)

    print("...Processing complete.")
    return pd.DataFrame(all_features), np.array(all_labels)


# --- 5. Main Pipeline (Corrected) ---

def run_clarity_ai_pipeline():
    """
    Main function to run the entire pipeline.
    (Corrected to fix SHAP IndexError and pass feature names)
    """
    
    # --- Download Data ---
    print("Downloading MIT-BIH Arrhythmia database from PhysioNet...")
    wfdb.dl_database('mitdb', dl_dir='mitdb')
    print("...Download complete.")
    
    # --- Load Data ---
    X, y = load_and_segment_data(RECORDS, use_binary_map=True) 
    
    # --- Preprocessing ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    
    print("\nOriginal class distribution (test set):")
    print(pd.Series(y_test).value_counts(normalize=True))
    
    # --- SMOTE Oversampling ---
    smote = SMOTE(random_state=42, sampling_strategy=0.7)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print("\nResampled class distribution (training set):")
    print(pd.Series(y_train_res).value_counts(normalize=True))
    
    # --- Model Training ---
    print("\nTraining LightGBM model...")
    # --- FIX: Pass feature names to LightGBM ---
    feature_names = X.columns.tolist()
    model = lgb.LGBMClassifier(
        n_estimators=1450,
        max_depth=11,
        learning_rate=0.05,
        num_leaves=38,
        subsample=0.85,
        colsample_bytree=0.75,
        objective='binary',
        metric='auc',
        n_jobs=-1,
        random_state=42
    )
    
    # Pass feature_name=feature_names
    model.fit(X_train_res, y_train_res, feature_name=feature_names)
    print("...Model training complete.")
    
    # --- Evaluation ---
    print("\nEvaluating model on test set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Classification Report (Test Set) ---")
    print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Anomaly (1)']))
    
    print("\n--- Generating Confusion Matrix (Figure 7) ---")
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Figure 7 (Binary): Normalized Confusion Matrix")
    plt.show()
    
    print("\n--- Generating ROC Curve (Figure 8) ---")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label=f"CLARITY-AI 2.0 (AUC = {auc:.3f})", color='blue', lw=2)
    plt.plot([0, 1], [0, 1], color='gray', ls='--', label='Random (AUC = 0.500)')
    plt.title("Figure 8: ROC Curve (MIT-BIH Test Set)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

    # --- XAI Generation (SHAP) ---
    print("\n--- Generating SHAP Explanations (Figures 11, 12) ---")
    
    # Create DataFrames with feature names for SHAP
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    X_test_subset = X_test_df.sample(500, random_state=42)
    
    explainer = shap.TreeExplainer(model)
    # --- FIX: shap_values is a list [shap_class_0, shap_class_1] ---
    shap_values_list = explainer.shap_values(X_test_subset)
    
    # Generate Global XAI Beeswarm (Figure 11)
    print("Plotting Figure 11: SHAP Beeswarm Plot...")
    # We plot the SHAP values for class 1 (Anomaly)
    shap.summary_plot(shap_values_list[1], X_test_subset, plot_type="dot", max_display=15, show=False)
    plt.title("Figure 11: Global XAI: SHAP Summary (Beeswarm) Plot")
    plt.tight_layout()
    plt.show()
    
    # --- FIX: Correctly handle list output for waterfall plot ---
    y_test_series = pd.Series(y_test, index=X_test_df.index)
    anomaly_indices = y_test_series[y_test_series == 1].index
    
    if len(anomaly_indices) > 0:
        # Find the index in the original X_test_df
        anomaly_index_loc = anomaly_indices[0] 
        # Select as a 1-row DataFrame
        anomaly_row_df = X_test_df.loc[[anomaly_index_loc]] 
        
        print("Plotting Figure 12: SHAP Waterfall Plot for one Anomaly...")
        
        # Get SHAP values for this single row
        # Output is still a list: [shap_class_0, shap_class_1]
        shap_values_single_list = explainer.shap_values(anomaly_row_df)
        
        # Get the expected value (base rate) for class 1
        # explainer.expected_value is also a list: [base_class_0, base_class_1]
        base_value_class_1 = explainer.expected_value[1]
        
        # Get the SHAP values for class 1, for the first (and only) row
        shap_values_class_1_single_row = shap_values_single_list[1][0]
        
        # Create the SHAP Explanation object
        shap.waterfall_plot(shap.Explanation(
            values=shap_values_class_1_single_row,
            base_values=base_value_class_1,
            data=anomaly_row_df.iloc[0],
            feature_names=feature_names
        ), max_display=10, show=False)
        
        plt.title("Figure 12: Local XAI: SHAP Waterfall (Anomaly)")
        plt.tight_layout()
        plt.show()
    else:
        print("No anomalies found in the test subset to plot Figure 12.")

# --- 6. Run the pipeline ---
if __name__ == "__main__":
    # Set SHAP js plotting for notebooks
    shap.initjs()
    run_clarity_ai_pipeline()
