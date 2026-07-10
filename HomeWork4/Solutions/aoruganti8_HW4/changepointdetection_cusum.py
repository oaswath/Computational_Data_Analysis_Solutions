import numpy as np
import matplotlib.pyplot as plt

# 1. Setup and Seed
np.random.seed(6740)

# 2. Generate Data
# First 100 samples from N(0, 1)
x_f0 = np.random.normal(0, np.sqrt(1.0), 100)
# Next 50 samples from N(0, 1.25)
x_f1 = np.random.normal(0, np.sqrt(1.25), 50)
x = np.concatenate((x_f0, x_f1))

# 3. Calculate CUSUM Statistic
S = np.zeros(len(x) + 1)
threshold_constant = 0.5 * np.log(1.25)

for t in range(1, len(x) + 1):
    W_t = 0.1 * (x[t-1]**2) - threshold_constant
    S[t] = max(0, S[t-1] + W_t)

# 4. Plotting
plt.figure(figsize=(10, 5))
plt.plot(range(len(S)), S, label='CUSUM Statistic (S_t)')
plt.axvline(x=100, color='r', linestyle='--', label='True Change Point (t=100)')
plt.title('CUSUM Statistic for Change in Variance')
plt.xlabel('Time (t)')
plt.ylabel('S_t')
plt.legend()
plt.grid(True)
plt.savefig('cusum_change_point_detection.png', bbox_inches='tight')