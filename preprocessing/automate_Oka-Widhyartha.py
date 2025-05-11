# Load libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# load the raw data
file_path = os.path.join(os.path.dirname(__file__), "..", "diabetes_raw.csv")
df_raw = pd.read_csv(file_path)

# print the first few rows of the raw data
print("Raw Data:")
print(df_raw.head())

# change data in gender column to uppercase
df_raw["Gender"] = df_raw["Gender"].str.upper()

# trim the CLASS column
df_raw["CLASS"] = df_raw["CLASS"].str.strip()

# encode Gender with pd.get_dummies and ensure values are 0 and 1
df_raw = pd.get_dummies(df_raw, columns=["Gender"], dtype=int)

# encode CLASS column with label encoding
le = LabelEncoder()
df_raw["CLASS"] = le.fit_transform(df_raw["CLASS"])

# get label encoded values and their corresponding classes and save to a dictionary
label_encoded_dict = {i: class_name for i, class_name in enumerate(le.classes_)}
print("Label Encoded Values:")
for i, class_name in label_encoded_dict.items():
    print(f"{i}: {class_name}")

# save label encoded values to a CSV file
label_encoded_df = pd.DataFrame(list(label_encoded_dict.items()), columns=["label", "class"])
label_encoded_file_path = os.path.join(os.path.dirname(__file__), "label_encoded_values.csv")
label_encoded_df.to_csv(label_encoded_file_path, index=False)
print("Label encoded values saved to label_encoded_values.csv")

# split the data into features and target
X = df_raw.drop(columns=["ID", "No_Pation", "CLASS"])
y = df_raw["CLASS"]

# split into train and test sets proportionally class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# standardize the numeric columns
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# create train and test dataframes
df_train = pd.DataFrame(X_train_scaled, columns=X.columns)
df_train["CLASS"] = y_train_resampled.values

df_test = pd.DataFrame(X_test_scaled, columns=X.columns)
df_test["CLASS"] = y_test.values

# check the first few rows df_train
print("First few rows of the final train set:")
print(df_train.head())

# check the first few rows df_test
print("First few rows of the final test set:")
print(df_test.head())

# save the train and test data to a new CSV file
output_dir = os.path.join(os.path.dirname(__file__), "diabetes_preprocessing")
os.makedirs(output_dir, exist_ok=True)

train_file_path = os.path.join(output_dir, "diabetes_train.csv")
test_file_path = os.path.join(output_dir, "diabetes_test.csv")

df_train.to_csv(train_file_path, index=False)
df_test.to_csv(test_file_path, index=False)
print("Train and test data saved to diabetes_preprocessing/diabetes_train.csv and diabetes_preprocessing/diabetes_test.csv")