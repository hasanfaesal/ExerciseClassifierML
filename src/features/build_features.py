import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans
# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")

predictor_columns = list(df.columns[:6])

# plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
df.info()
df.isna().sum()

for col in predictor_columns:
    df[col] = df[col].interpolate()

df.info()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

df[df["set"] == 25]["acc_y"].plot()
df[df["set"] == 51]["acc_y"].plot()

duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]
duration.seconds

for s in df["set"].unique():

    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]

    duration = stop - start
    df.loc[(df["set"] == s), "duration"] = duration.seconds

duration_df = df.groupby(["volume"])["duration"].mean()

duration_df.iloc[0] / 5
duration_df.iloc[1] / 10

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
LowPass = LowPassFilter()

fs = 1000/200
cutoff = 1.2

df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)


for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]
    
     

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()
pca = PrincipalComponentAnalysis()

pc_values = pca.determine_pc_explained_variance(df_pca, predictor_columns)

df_pca = pca.apply_pca(df_pca, predictor_columns, 3)

subset = df_pca[df_pca["set"] == 35]
subset[["pca_1","pca_2","pca_3"]].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------
df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

ws = int(1000 / 200)

for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std")

df_temporal_list = []

for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
    df_temporal_list.append(subset)
    
df_temporal = pd.concat(df_temporal_list)

df_temporal.isna().sum()
df_temporal.info()

subset[["acc_y","acc_y_temp_mean_ws_5","acc_y_temp_std_ws_5"]].plot()
subset[["gyr_y","gyr_y_temp_mean_ws_5","gyr_y_temp_std_ws_5"]].plot()
#IN MY OPINION TOO MUCH SMOOTH? WS VALUE SHOULD BE LESS LIKE 800 MS?

# --------------------------------------------------------------
# Frequency features, Discrete Fourier Transformation
# --------------------------------------------------------------

df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

fs = int(1000/200)
ws = int(2800/200)

df_freq = FreqAbs.abstract_frequency(df_freq,["acc_y"], ws, fs)

df_freq.columns

df_freq_list = []
for s in df_freq["set"].unique():
    print(f"Applying Fourier Transformation to set {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)
    
df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

df_freq = df_freq.dropna()

df_freq = df_freq.iloc[::2]
# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

df_cluster = df_freq.copy()

cluster_columns = ["acc_x","acc_y","acc_z"]
k_values = range(2,10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)
    
plt.figure(figsize=(10,10))
plt.plot(k_values,inertias)
plt.xlabel("x")
plt.ylabel("SS")
plt.show()

kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

#plot clusters
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

#plot accelerometer data to compare
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection="3d")
for l in df_cluster["exercise"].unique():
    subset = df_cluster[df_cluster["exercise"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=l)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")