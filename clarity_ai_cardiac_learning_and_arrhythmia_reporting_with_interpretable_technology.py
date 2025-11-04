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
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

# Suppress ignorable warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


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
RECORDS = sorted(list(set(RECORDS))) # Ensure unique records

# --- 3. Feature Extraction (Corrected & Robust) ---

def extract_hybrid_features(segment, rr_intervals_samples, fs=360):
    """
    Extracts the hybrid features from a single ECG beat segment.
    (This version is robust against feature extraction failures and correctly
     calculates QRS duration.)
    """
    features = {}
    
    # 1. Initialize ALL 24 features to 0.0. This guarantees a consistent shape.
    feature_names = [
        'mean', 'std', 'ptp', 'min', 'max', 'mad', # 6 stats
        'qrs_duration', 'hrv_sdnn', 'hrv_rmssd', 'hrv_lf_hf_ratio', # 4 hrv
        'poincare_sd1', 'poincare_sd2', # 2 non-linear
        'rr_interval_post', 'rr_interval_pre' # 2 rr
    ]
    for f in feature_names:
        features[f] = 0.0
    # level=4 produces 5 coefficient arrays: [cA4, cD4, cD3, cD2, cD1]
    # 5 levels * 2 features (energy, entropy) = 10 wavelet features
    for i in range(5): 
        features[f'wavelet_energy_L{i}'] = 0.0
        features[f'wavelet_entropy_L{i}'] = 0.0
            
    # Now, try to fill them in.
    
    # 2. Statistical Features (Safe)
    features['mean'] = np.mean(segment)
    features['std'] = np.std(segment)
    features['ptp'] = np.ptp(segment)
    features['min'] = np.min(segment)
    features['max'] = np.max(segment)
    features['mad'] = np.mean(np.abs(segment - features['mean']))

    # 3. Wavelet Features (Try/Except)
    try:
        if np.all(np.isfinite(segment)):
            coeffs = pywt.wavedec(segment, 'db4', level=4)
            for i, c in enumerate(coeffs):
                c_finite = c[np.isfinite(c)]
                if len(c_finite) > 0:
                    features[f'wavelet_energy_L{i}'] = np.sum(c_finite**2)
                    features[f'wavelet_entropy_L{i}'] = pywt.shannon_entropy(c_finite)
    except Exception:
        pass # Will keep zeros

    # 4. Fiducial, HRV, Non-Linear (Granular Try/Except)
    try:
        # --- *** CORRECTED QRS DURATION *** ---
        # The R-peak is at the center of the segment (e.g., sample 128)
        r_peak_in_segment = [len(segment) // 2]
        try:
            # Delineate just this single beat
            _, waves = nk.ecg_delineate(segment, r_peak_in_segment, sampling_rate=fs, method="peak")
            
            # Check if delineation was successful and found Q and S peaks
            if waves['ECG_Q_Peaks'] and pd.notna(waves['ECG_Q_Peaks'][0]) and \
               waves['ECG_S_Peaks'] and pd.notna(waves['ECG_S_Peaks'][0]):
                
                qrs_onset = waves['ECG_Q_Peaks'][0]
                qrs_offset = waves['ECG_S_Peaks'][0]
                
                if qrs_offset > qrs_onset:
                    duration_ms = ((qrs_offset - qrs_onset) / fs) * 1000
                    # Plausibility check for a valid QRS duration
                    if duration_ms > 0 and duration_ms < 250: 
                         features['qrs_duration'] = duration_ms
        except Exception:
            pass # Keep qrs_duration as 0
        # --- *** END OF QRS CORRECTION *** ---
            
        rr_intervals_ms = (rr_intervals_samples / fs) * 1000
        rr_intervals_ms = rr_intervals_ms[np.isfinite(rr_intervals_ms) & (rr_intervals_ms > 0)]
        
        if len(rr_intervals_ms) > 3: # Need at least 4 RRs for stable HRV
            try:
                hrv_time = nk.hrv_time(rr_intervals_ms, sampling_rate=1000)
                features['hrv_sdnn'] = hrv_time['HRV_SDNN'].iloc[0]
                features['hrv_rmssd'] = hrv_time['HRV_RMSSD'].iloc[0]
            except Exception:
                pass # Keep 0
            
            try:
                hrv_freq = nk.hrv_frequency(rr_intervals_ms, sampling_rate=1000, vlf=False, hf=True, lf=True)
                features['hrv_lf_hf_ratio'] = hrv_freq['HRV_LFHF'].iloc[0]
            except Exception:
                pass # Keep 0
            
            try:
                hrv_nl = nk.hrv_nonlinear(rr_intervals_ms, sampling_rate=1000)
                features['poincare_sd1'] = hrv_nl['HRV_SD1'].iloc[0]
                features['poincare_sd2'] = hrv_nl['HRV_SD2'].iloc[0]
            except Exception:
                pass # Keep 0
    except Exception:
        pass # Keep all HRV as 0
            
    # 5. RR Features (Safe)
    if len(rr_intervals_samples) > 0:
        features['rr_interval_post'] = rr_intervals_samples[-1]
    if len(rr_intervals_samples) > 1:
        features['rr_interval_pre'] = rr_intervals_samples[-2]

    return pd.Series(features).fillna(0.0)

# --- 4. Data Loading ---

def load_and_segment_data(records, window_size=256, use_binary_map=True):
    """
    Loads data from wfdb, segments beats, and extracts features.
    """
    all_features = []
    all_labels = []
    
    print(f"Loading and processing {len(records)} records...")
    
    for rec_name in records:
        print(f"Processing record: {rec_name}")
        
        try:
            record = wfdb.rdrecord(f'mitdb/{rec_name}', sampto=30*60*360) # 30 min
            annotation = wfdb.rdann(f'mitdb/{rec_name}', 'atr', sampto=30*60*360)
        except Exception as e:
            print(f"Failed to read record {rec_name}: {e}")
            continue
            
        signal = record.p_signal[:, 0] # Use lead 1
        r_peaks = annotation.sample
        symbols = annotation.symbol
        
        # Map labels
        if use_binary_map:
            labels = [binary_annotation_map.get(s) for s in symbols]
        else:
            # You would define multi_class_map if use_binary_map=False
            labels = [multi_class_map.get(s) for s in symbols] 
        
        half_window = window_size // 2
        
        for i in range(1, len(r_peaks) - 1): # Skip first and last beat
            label = labels[i]
            if label is None:
                continue
                
            start = r_peaks[i] - half_window
            end = r_peaks[i] + half_window
            if start < 0 or end > len(signal):
                continue
                
            segment = signal[start:end]
            
            # Get R-peaks *before* the current one for HRV context
            rr_context_indices = np.where(r_peaks < r_peaks[i])[0]
            if len(rr_context_indices) < 10: # Need some history
                continue
            
            # Use the last 30 R-peaks for a stable HRV
            context_indices = rr_context_indices[-30:] 
            last_rrs = np.diff(r_peaks[context_indices]) 
            
            features = extract_hybrid_features(segment, last_rrs, fs=record.fs)
            all_features.append(features)
            all_labels.append(label)

    print("...Processing complete.")
    return pd.DataFrame(all_features), np.array(all_labels)


# --- 5. Main Pipeline (Corrected) ---

def run_clarity_ai_pipeline():
    """
    Main function to run the entire pipeline.
    (v10: Fixes QRS duration and SHAP API)
    """
    
    # --- Download Data ---
    print("Downloading MIT-BIH Arrhythmia database from PhysioNet...")
    wfdb.dl_database('mitdb', dl_dir='mitdb')
    print("...Download complete.")
    
    # --- Load Data ---
    X, y = load_and_segment_data(RECORDS, use_binary_map=True) 
    
    # --- Drop any rows that are all-zero (from failed extraction)
    # And align y with the new index of X
    X = X.loc[(X != 0).any(axis=1)]
    y = y[X.index]
    
    # --- Filter out zero-variance features ---
    # This is critical for SHAP compatibility
    variance = X.var()
    non_constant_features = variance[variance > 1e-4].index.tolist()
    X = X[non_constant_features]
    
    print(f"\n--- Feature extraction successful. ---")
    print(f"Total features with variance used for model: {len(non_constant_features)}")
    print(f"Total beats for training/testing: {X.shape[0]}\n")
    # ---
    
    # --- Preprocessing ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # This list now only contains the features the model will *actually* see.
    feature_names = X.columns.tolist() 
    
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

    # --- XAI Generation (SHAP) - CORRECTED API ---
    print("\n--- Generating SHAP Explanations (Figures 11, 12) ---")
    
    # Create DF from the scaled test data
    X_test_df = pd.DataFrame(X_test, columns=feature_names) 
    
    # We sample a smaller subset for faster global plot generation
    X_test_subset = X_test_df.sample(200, random_state=42, replace=True)
    
    # 1. Use shap.TreeExplainer
    explainer = shap.TreeExplainer(model)
    
    # 2. *** NEW API CALL ***
    # Use explainer(X) instead of explainer.shap_values(X).
    # This returns a single, powerful 'Explanation' object.
    # The shape will be (n_samples, n_features, n_classes)
    shap_values = explainer(X_test_subset)

    # Generate Global XAI Beeswarm (Figure 11)
    print("Plotting Figure 11: SHAP Beeswarm Plot...")
    
    # 3. *** NEW PLOTTING CALL ***
    # We pass the Explanation object directly, slicing it for class 1.
    # shap_values[..., 1] means [all_samples, all_features, class_1]
    # 3. *** NEW PLOTTING CALL ***
# We pass the entire Explanation object, which already represents class 1.
    shap.summary_plot(shap_values, X_test_subset, plot_type="dot", max_display=15, show=False)
    
    plt.title("Figure 11: Global XAI: SHAP Summary (Beeswarm) Plot")
    plt.tight_layout()
    plt.show()
    
    # 4. Find an anomaly to plot
    # Re-create y_test_series with the same index as X_test_df
    y_test_series = pd.Series(y_test, index=X_test_df.index)
    anomaly_indices = y_test_series[y_test_series == 1].index
    
    if len(anomaly_indices) > 0:
        # Get the row index (e.g., 1054)
        anomaly_index_loc = anomaly_indices[0] 
        # Select that one row from the *original* X_test_df
        anomaly_row_df = X_test_df.loc[[anomaly_index_loc]] 
        
        print("Plotting Figure 12: SHAP Waterfall Plot for one Anomaly...")
        
        # 5. *** NEW API CALL ***
        # Get the explanation for just that single row
        shap_values_single = explainer(anomaly_row_df)
        
        # 6. *** NEW PLOTTING CALL ***
        # Pass the sliced Explanation object for the first sample (0) and class 1
        # shap_values_single[0, ..., 1] = [sample_0, all_features, class_1]
        # 6. *** NEW PLOTTING CALL ***
# Pass the Explanation object for the first sample (index 0).
# This object has shape (n_features,) and is what waterfall_plot expects.
        shap.waterfall_plot(shap_values_single[0], max_display=10, show=False)
        
        plt.title("Figure 12: Local XAI: SHAP Waterfall (Anomaly)")
        plt.tight_layout()
        plt.show()
    else:
        print("No anomalies found in the test subset to plot Figure 12.")

# --- 6. Run the pipeline (Main Execution) ---
if __name__ == "__main__":
    # Set SHAP js plotting for notebooks
    shap.initjs()
    run_clarity_ai_pipeline()
