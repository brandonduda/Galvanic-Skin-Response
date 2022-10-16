import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

signalGSR = pd.read_csv('HRGSR/324230.csv', usecols=[1])
GSR = signalGSR.to_numpy()
GSR = GSR.reshape(-1)
smoothGSR = savgol_filter(GSR, 2001, 3)
t = 300
subGSR = [sum(smoothGSR[i:i + t])/t for i in range(len(smoothGSR) - t + 1)]
grad = abs(np.gradient(subGSR))
arr = np.array([])
for i in range(0, len(grad), 300):
    if grad[i] >= 0.025:
        valid_indices = np.append(arr, i)
        print("Sweat Increase at times (min)", valid_indices / 600)
    else:
        invalid_indices = np.append(arr, i)
        print("No Sweat Increase at times (min)", invalid_indices / 600)

plt.plot(GSR, label='Raw Data')
plt.plot(subGSR, label='Clean Data', linewidth=2)
x_min = 0
x_max = len(GSR) + 1
x_step = int(len(GSR) / (x_max/600))
plt.xticks(np.arange(x_min, x_max, x_step), np.arange(x_min, x_max/600))
plt.title('GSR')
plt.xlabel('Time (minutes)')
plt.ylabel('Analog')
plt.legend()
plt.show()


