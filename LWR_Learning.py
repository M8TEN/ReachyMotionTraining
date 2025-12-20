import numpy as np

class LWR:
    def learn_weights(self, demo_force, phase, kernels=50):
        c = np.linspace(1, 0, kernels)
        h = kernels**2 / (c**2 + 1e-10)

        weights = np.zeros(kernels)

        for i in range(kernels):
            psi = np.exp(-h[i] * (phase - c[i])**2)
            numerator = np.sum(psi * demo_force * phase)
            denominator = np.sum(psi * (phase**2))
            try:
                weights[i] = numerator/denominator
            except ZeroDivisionError:
                weights[i] = numerator / (denominator + 1e-10)
    
        return weights