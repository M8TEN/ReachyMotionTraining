import numpy as np
import matplotlib.pyplot as plt

TAU = 1.0
SAMPLE_RATE = 1/100.0

def canonical_system(alpha_x, steps=1_000, start=1):
    phase_values = []
    x = start
    for i in range(steps):
        x_prime = (-alpha_x*x) / TAU
        x += x_prime * SAMPLE_RATE
        phase_values.append(x)
    
    return np.array(phase_values)

x_values = list(range(1000))

for alpha in range(1,6):
    plt.plot(x_values, canonical_system(alpha), label=f"Alpha = {alpha}")

plt.legend()
plt.show()