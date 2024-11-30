import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np

df = pd.read_json('../data/extracted_data.jsonl', lines=True)
df['appeared'] = pd.to_datetime(df['appeared'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

df = df[(df['timestamp'] >= dt.datetime(2016, 1, 1)) & (df['timestamp'] <= dt.datetime(2019, 1, 1))]
df = df[df['appeared'] >= dt.datetime(2017, 1, 1)]
print(df['timestamp'].count())

plt.scatter(df['appeared'][df['label'] == 0], df['timestamp'][df['label'] == 0], s=1, c='b', label='benign', alpha=0.005)
plt.scatter(df['appeared'][df['label'] == 1], df['timestamp'][df['label'] == 1], s=1, c='r', label='mal', alpha=0.005)
plt.ylim((dt.datetime(2017, 1, 1), dt.datetime(2019, 1, 1)))
plt.legend()
plt.show()

month_equal = np.abs(df['appeared'] - df['timestamp']) <= dt.timedelta(days=365*2)
print(df['timestamp'][month_equal].count())