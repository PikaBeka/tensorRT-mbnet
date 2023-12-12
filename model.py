import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import sys

df = pd.read_csv('./dataset.csv')
df = df.drop_duplicates()
df = df.reset_index(drop=True)

tensorRT = df[df["Fastest Method"] == "TensorRT"]
cublass  = df[df["Fastest Method"] == "CUBLASS"]
cudnn  = df[df["Fastest Method"] == "CUDNN_OPT"]


cudnn_upsample = resample(cudnn,
             replace=True,
             n_samples=len(tensorRT),
             random_state=42)

cublass_upsample = resample(cublass,
             replace=True,
             n_samples=len(tensorRT),
             random_state=42)

data_upsampled = pd.concat([tensorRT, cudnn_upsample, cublass_upsample])

headers = data_upsampled.columns
X_train = data_upsampled[headers[:-2]]
y_train = data_upsampled[headers[-2]]

le = LabelEncoder()
y_train = le.fit_transform(y_train)

model = RandomForestClassifier(random_state=2023, criterion='gini', max_depth = 8, min_samples_leaf=1, min_samples_split=8, n_estimators = 15)
model.fit(X_train, y_train)

C = int(sys.argv[1])
HW = int(sys.argv[2])
K = int(sys.argv[3])
RS = int(sys.argv[4])

x = np.array([[C, HW, K, RS]])

pred = model.predict(x)

if pred[0] == 0:
	print("CUBLASS")
if pred[0] == 1:
	print("CUDNN_OPT")
if pred[0] == 2:
	print("TensorRT")
