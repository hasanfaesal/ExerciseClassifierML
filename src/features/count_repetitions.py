import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")
df = df[df["exercise"] != "rest"]
# ss

acc_r = df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"]
gyr_r = df["gyr_x"] ** 2 + df["gyr_y"] ** 2 + df["gyr_z"]

df["acc_r"] = np.sqrt(acc_r)
df["gyr_r"] = np.sqrt(gyr_r)
# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------

bench_df = df[df["exercise"] == "bench"]
squat_df = df[df["exercise"] == "squat"]
row_df = df[df["exercise"] == "row"]
ohp_df = df[df["exercise"] == "ohp"]
dead_df = df[df["exercise"] == "dead"]

# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------

plot_df = squat_df

plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_r"].plot()

plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_r"].plot()

# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------

fs = 1000 / 200
LowPass = LowPassFilter()

# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------

bench_set = bench_df[bench_df["set"] == bench_df["set"].unique()[0]]
squat_set = squat_df[squat_df["set"] == squat_df["set"].unique()[0]]
row_set = row_df[row_df["set"] == row_df["set"].unique()[0]]
ohp_set = ohp_df[ohp_df["set"] == ohp_df["set"].unique()[0]]
dead_set = dead_df[dead_df["set"] == dead_df["set"].unique()[0]]

bench_set["acc_r"].plot()

column = "acc_r"
LowPass.low_pass_filter(
    ohp_set, col=column, sampling_frequency=fs, cutoff_frequency=0.4, order=10
)[column + "_lowpass"].plot()
# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------
column = "acc_r"
data = LowPass.low_pass_filter(
    bench_set, col=column, sampling_frequency=fs, cutoff_frequency=0.4, order=10
)[column + "_lowpass"]

data["acc_r_lowpass"]
argrelextrema(data[column + "_lowpass"].values, np.greater)

# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------


# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------
