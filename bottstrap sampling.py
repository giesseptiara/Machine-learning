import numpy as np
# Data
data = np.array([10, 15, 8, 12, 14, 20, 18, 16, 11, 13])
# Bootstrap sampling
n_samples = 1000
bootstrap_means = np.zeros(n_samples)
for i in range(n_samples): bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
bootstrap_means[i] = np.mean(bootstrap_sample)
# Confidence interval (95%)
ci_lower = np.percentile(bootstrap_means, 2.5)
ci_upper = np.percentile(bootstrap_means, 97.5)
print("Mean:", np.mean(data))
print("95% Confidence Interval:", ci_lower, "-", ci_upper)