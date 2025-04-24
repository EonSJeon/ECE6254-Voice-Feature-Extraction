import os
import gc
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm  # 로딩바용

# ------------------------------------------------------------------------
# 1) Filtering & Sampling (Train/Valid/Test)
# ------------------------------------------------------------------------
ACCENTS_OF_INTEREST = ['us', 'england', 'indian', 'australia']
BASE_DIR = '/Users/jeonsang-eon/ECE6254-Voice-Feature-Extraction/'
RAW_DATA_DIR = os.path.join(BASE_DIR, 'raw_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_filter_csv(csv_path, accents=ACCENTS_OF_INTEREST):
    """
    1) CSV 로드
    2) gender, accent 결측 제거
    3) 관심 액센트만 남김
    4) 실제 오디오 파일 존재 여부 체크
    5) 필요한 컬럼만 추출 (filename, text, gender, accent, age)
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['gender', 'accent'])
    df = df[df['accent'].isin(accents)]
    df = df[df['filename'].apply(lambda x: os.path.exists(os.path.join(RAW_DATA_DIR, x)))]
    keep_cols = ['filename', 'text', 'gender', 'accent', 'age']
    df = df[keep_cols].copy()
    return df

def balance_by_gender_age(df):
    """
    (train 전용) (gender, age) 조합별로 모든 accent가 균등해지도록
    오버샘플링 또는 언더샘플링
    """
    df_temp = df.copy()
    all_accents = df_temp['accent'].unique()
    num_accents = len(all_accents)
    
    combo_counts = df_temp.groupby(['gender', 'age'])['accent'].nunique()
    valid_combos = combo_counts[combo_counts == num_accents].index.tolist()
    
    balanced_groups = []
    resample_info = []
    
    # 각 (gender, age) 조합마다 진행 상황을 로깅 (tqdm 사용 가능)
    for gender, age in tqdm(valid_combos, desc="Balancing by (gender, age)"):
        subset = df_temp[(df_temp['gender'] == gender) & (df_temp['age'] == age)]
        accent_counts = subset.groupby('accent').size()
        
        T = min(accent_counts.max(), int(1.5 * accent_counts.min()))
        
        for accent_val, group in subset.groupby('accent'):
            current_count = len(group)
            factor = T / current_count
            if current_count < T:
                sampled = group.sample(n=T, replace=True, random_state=42)
                method = 'oversampled'
            else:
                sampled = group.sample(n=T, replace=False, random_state=42)
                method = 'undersampled'
            
            balanced_groups.append(sampled)
            info_line = (f"[{gender}, {age}, {accent_val}]  "
                         f"Original={current_count}, Target={T}, "
                         f"Factor={factor:.2f}, Method={method}")
            resample_info.append(info_line)

    df_balanced = pd.concat(balanced_groups, ignore_index=True)
    return df_balanced, resample_info

def split_train_valid_by_accent(df_balanced, df_train_original, valid_size=500):
    """
    df_balanced 에 포함되지 않은 나머지 샘플들 중에서
    accent별 최대 valid_size개를 샘플링하여 df_valid를 생성
    """
    df_candidates = df_train_original[~df_train_original['filename'].isin(df_balanced['filename'])]
    valid_groups = []
    for accent, group in df_candidates.groupby('accent'):
        n_samples = min(valid_size, len(group))
        sampled_group = group.sample(n=n_samples, random_state=42)
        valid_groups.append(sampled_group)
    df_valid = pd.concat(valid_groups, ignore_index=True)
    return df_valid

# ------------------------------------------------------------------------
# 2) Feature Extraction to NPZ (오디오 + 텍스트)
# ------------------------------------------------------------------------
import librosa
from scipy.signal import butter, lfilter
from transformers import BertTokenizer, BertModel
from tensorflow.keras.utils import to_categorical

def lowpass_filter(data, sr, cutoff=4000, order=5):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def extract_mfcc(file_path, n_mfcc=13, cutoff=5000):
    audio, sr = librosa.load(file_path, sr=None)
    audio = lowpass_filter(audio, sr, cutoff=cutoff)
    win_length = int(0.025 * sr)
    hop_length = int(0.01 * sr)
    n_fft = win_length
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window='hann'
    )
    return mfcc  # shape: (n_mfcc, T)

# BERT 토크나이저 로드 (원하는 모델로 교체 가능)
BERT_MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

def tokenize_text(text, max_length=32):
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='np'
    )
    input_ids = encoding['input_ids'][0]
    attention_mask = encoding['attention_mask'][0]
    return input_ids, attention_mask

def pad_or_truncate_mfcc(mfcc, target_time_frames):
    n_mfcc, T = mfcc.shape
    if T < target_time_frames:
        padded = np.pad(mfcc, ((0, 0), (0, target_time_frames - T)), mode='constant')
    else:
        padded = mfcc[:, :target_time_frames]
    return padded

def create_npz_from_csv(
    csv_path,
    audio_base_dir,
    output_npz_path,
    n_mfcc=13,
    cutoff=5000,
    max_length_text=32,
    time_frames_target=None,
    label_col='accent_encoded'
):
    df = pd.read_csv(csv_path)
    
    mfcc_list = []
    input_ids_list = []
    attention_mask_list = []

    print(f"\n[create_npz_from_csv] Processing {len(df)} rows from {csv_path}")
    # tqdm을 사용하여 진행 상황 표시
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        audio_path = os.path.join(audio_base_dir, row['filename'])
        mfcc_data = extract_mfcc(audio_path, n_mfcc=n_mfcc, cutoff=cutoff)
        mfcc_list.append(mfcc_data)
        
        txt = str(row.get('text', ''))
        ids, mask = tokenize_text(txt, max_length=max_length_text)
        input_ids_list.append(ids)
        attention_mask_list.append(mask)
    
    # time_frames_target이 None이면 median으로 계산
    if time_frames_target is None:
        all_lengths = [m.shape[1] for m in mfcc_list]
        time_frames_target = int(np.median(all_lengths))
        print(f"  -> Using computed median time_frames={time_frames_target}")
    else:
        print(f"  -> Using provided time_frames={time_frames_target}")
    
    padded_mfcc_list = [pad_or_truncate_mfcc(m, time_frames_target) for m in mfcc_list]
    X_audio = np.stack([np.expand_dims(m, axis=0) for m in padded_mfcc_list])
    
    X_input_ids = np.stack(input_ids_list)
    X_attention_mask = np.stack(attention_mask_list)
    
    labels = df[label_col].values
    y = to_categorical(labels)
    
    np.savez_compressed(
        output_npz_path,
        X_audio=X_audio,
        X_input_ids=X_input_ids,
        X_attention_mask=X_attention_mask,
        y=y,
        time_frames=time_frames_target
    )
    print(f"  -> Saved to {output_npz_path}")
    print(f"  -> X_audio.shape={X_audio.shape}, y.shape={y.shape}")
    
    # 반환값으로 time_frames_target도 함께 반환하면 후속 처리에 활용 가능
    return time_frames_target

# ------------------------------------------------------------------------
# 3) 실행 (main)
# ------------------------------------------------------------------------
if __name__ == "__main__":
    # -------------------
    # (A) Train CSV 로드 & 필터링
    TRAIN_CSV_ORIG = os.path.join(RAW_DATA_DIR, "cv-valid-train.csv")
    df_train_filtered = load_and_filter_csv(TRAIN_CSV_ORIG, accents=ACCENTS_OF_INTEREST)
    print("\nFiltered train accent counts:")
    print(df_train_filtered['accent'].value_counts())

    # (B) Balance train
    df_train_balanced, info_lines = balance_by_gender_age(df_train_filtered)
    info_txt_path = os.path.join(OUTPUT_DIR, "train_dataset_info.txt")
    with open(info_txt_path, "w") as f:
        f.write("Resample Factor Information:\n")
        for line in info_lines:
            f.write(line + "\n")
    print("\n[train_dataset_info] saved to", info_txt_path)

    # (C) Split valid
    df_valid = split_train_valid_by_accent(df_train_balanced, df_train_filtered, valid_size=500)
    print("df_train_balanced shape:", df_train_balanced.shape)
    print("df_valid shape:", df_valid.shape)

    # (D) Test CSV 로드 & 필터링
    TEST_CSV_ORIG = os.path.join(RAW_DATA_DIR, "cv-valid-test.csv")
    df_test_filtered = load_and_filter_csv(TEST_CSV_ORIG, accents=ACCENTS_OF_INTEREST)
    print("\nFiltered test accent counts:")
    print(df_test_filtered['accent'].value_counts())

    # (E) Label Encoding (train_balanced -> fit, valid/test -> transform)
    label_encoder = LabelEncoder()
    df_train_balanced['accent_encoded'] = label_encoder.fit_transform(df_train_balanced['accent'])
    df_valid['accent_encoded'] = label_encoder.transform(df_valid['accent'])
    df_test_filtered['accent_encoded'] = label_encoder.transform(df_test_filtered['accent'])

    label_map_path = os.path.join(OUTPUT_DIR, "label_mapping_info.txt")
    with open(label_map_path, "w") as f:
        f.write("Accent Label Mapping:\n")
        for idx, accent in enumerate(label_encoder.classes_):
            f.write(f"{idx}: {accent}\n")
    print("\nLabel mapping saved to", label_map_path)

    # (F) CSV 저장
    train_csv_balanced_path = os.path.join(OUTPUT_DIR, "df_train_balanced.csv")
    valid_csv_path = os.path.join(OUTPUT_DIR, "df_valid.csv")
    test_csv_path = os.path.join(OUTPUT_DIR, "df_test.csv")

    df_train_balanced.to_csv(train_csv_balanced_path, index=False)
    df_valid.to_csv(valid_csv_path, index=False)
    df_test_filtered.to_csv(test_csv_path, index=False)

    print(f"\nSaved train_balanced.csv -> {train_csv_balanced_path}")
    print(f"Saved valid.csv -> {valid_csv_path}")
    print(f"Saved test.csv -> {test_csv_path}")

    # -------------------
    # (G) Feature Extraction & NPZ
    # -------------------

    # Train NPZ 생성 시
    train_npz_path = os.path.join(OUTPUT_DIR, "train_dataset.npz")
    train_time_frames = create_npz_from_csv(
        csv_path=train_csv_balanced_path,
        audio_base_dir=RAW_DATA_DIR,
        output_npz_path=train_npz_path,
        n_mfcc=20,
        cutoff=5000,
        max_length_text=32,
        time_frames_target=None,  # median 계산
        label_col='accent_encoded'
    )
        
    # Valid NPZ 생성 시, train의 median time_frames를 사용
    valid_npz_path = os.path.join(OUTPUT_DIR, "valid_dataset.npz")
    create_npz_from_csv(
        csv_path=valid_csv_path,
        audio_base_dir=RAW_DATA_DIR,
        output_npz_path=valid_npz_path,
        n_mfcc=20,
        cutoff=5000,
        max_length_text=32,
        time_frames_target=train_time_frames,  # 동일 값 사용
        label_col='accent_encoded'
    )

    # Test NPZ 생성 시, 역시 동일 값을 사용
    test_npz_path = os.path.join(OUTPUT_DIR, "test_dataset.npz")
    create_npz_from_csv(
        csv_path=test_csv_path,
        audio_base_dir=RAW_DATA_DIR,
        output_npz_path=test_npz_path,
        n_mfcc=20,
        cutoff=5000,
        max_length_text=32,
        time_frames_target=train_time_frames,  # 동일 값 사용
        label_col='accent_encoded'
    )

    gc.collect()