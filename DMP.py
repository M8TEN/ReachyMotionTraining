import numpy as np
from scipy.signal import savgol_filter
from math import exp as mexp

joint_names = [
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_arm_yaw",
    "l_elbow_pitch",
    "l_forearm_yaw",
    "l_wrist_pitch",
    "l_wrist_roll",
    "l_gripper",
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_arm_yaw",
    "r_elbow_pitch",
    "r_forearm_yaw",
    "r_wrist_pitch",
    "r_wrist_roll",
    "r_gripper",
    "neck_pitch",
    "neck_roll",
    "neck_yaw"
]

class DMP():
    def __init__(self, sample_rate: float, tau: float, alpha_z: float, alpha_x: float, joint_path: np.ndarray, phase: np.ndarray):
        self.sample_rate = sample_rate
        self.alpha_z = alpha_z
        self.beta_z = alpha_z/4.0
        self.alpha_x = alpha_x
        self.tau = tau
        self.phase = phase
        self.smooth_samples = savgol_filter(joint_path, 59, 3)
        self.velocity = savgol_filter(joint_path, 59, 3, 1, self.sample_rate)
        self.acceleration = savgol_filter(joint_path, 59, 3, 2, self.sample_rate)
        self.weights = np.array([])
        self.kernel_centers = np.array([])
        self.kernel_widths = np.array([])
    
    def calculate_f_target(self, y, ydot, yddot, g):
        f_target = self.tau**2 * yddot - self.alpha_z * (self.beta_z * (g-y) - self.tau*ydot)
        return f_target

    def basis_function(self, center, width, x):
        return mexp(-width * (x - center)**2)

    def forcing_function(self, kernels, widths, weights, phase, start, goal) -> float:
        numerator = 0
        denominator = 0
        for i in range(len(kernels)):
            basis = self.basis_function(kernels[i], widths[i], phase)
            numerator += basis * weights[i]
            denominator += basis
        
        if abs(denominator) < 1e-10:
            return 0

        return numerator/denominator * phase * (goal - start)

    def learn_weights(self) -> None:
        start = self.smooth_samples[0]
        goal = self.smooth_samples[-1]
        f_target = self.calculate_f_target(self.smooth_samples, self.velocity, self.acceleration, goal)

        # Learn weights
        N: int = 50 #Number of Basis functions
        time_distribution = np.linspace(0, self.tau, N)
        self.kernel_centers = np.exp(-self.alpha_x/self.tau*time_distribution)
        self.kernel_widths = np.zeros(N)

        for i in range(N-1):
            distance = self.kernel_centers[i] - self.kernel_centers[i+1]
            self.kernel_widths[i] = 1 / ((0.55 * (distance))**2)

        self.kernel_widths[-1] = self.kernel_widths[-2]
        weights = []

        # w[i] = (s.transposed * T[i] * f_target) / (s.transposed * T[i] * s)
        for i in range(N):
            s: np.ndarray = self.phase * (goal - start) # Since Eta = x(t)(g-y_0)
            s_trans = s.transpose()
            T_i = np.diag(np.array([self.basis_function(self.kernel_centers[i], self.kernel_widths[i], x) for x in self.phase]))
            w_i = (s_trans @ T_i @ f_target) / (s_trans @ T_i @ s)
            weights.append(w_i)
        
        self.weights = np.array(weights)

    def produce_movement(self, start: float, goal: float, new_tau: float, time_vector: np.ndarray) -> np.ndarray:
        # Reproduce the movement using learned weights
        new_x = 1
        position = self.smooth_samples[0]
        vel = self.velocity[0]
        reproduced_movement = []

        for t in time_vector:
            f = self.forcing_function(self.kernel_centers, self.kernel_widths, self.weights, new_x, start, goal)
            dx = -self.alpha_x*new_x / new_tau
            new_x += dx * self.sample_rate
            dz = (self.alpha_z * (self.beta_z * (goal - position) - vel) + f) / new_tau
            dy = vel / new_tau
            vel += dz * self.sample_rate
            position += dy * self.sample_rate
            reproduced_movement.append(position)
        
        return np.array(reproduced_movement)

def create_phase_vector(alpha_x, tau, sample_rate, time_steps: int):
    x = 1
    phase_values = [x]
    for i in range(time_steps-1):
        x_prime = -alpha_x*x / tau
        x += x_prime * sample_rate
        phase_values.append(x)
    return np.array(phase_values)

if __name__ == "__main__":
    import pickle
    import matplotlib.pyplot as plt
    FILE_PATH: str = "Recordings/JointSamples7.pkl"
    NUMBER_OF_JOINTS: int = 19
    SAMPLE_RATE: float = 1/60.0
    ALPHA_X: float = 25
    ALPHA_Z: float = 50
    BETA_Z: float = ALPHA_Z/4.0


    # Load in recording data
    with open(FILE_PATH, "rb") as file:
        all_samples = pickle.load(file)

    TAU = len(all_samples)/NUMBER_OF_JOINTS*SAMPLE_RATE
    num_of_samples = int(len(all_samples)/NUMBER_OF_JOINTS)
    time_space = np.linspace(0, TAU, num_of_samples)
    joint_dmps = []

    ROWS = 4
    COLUMNS = 5

    for i in range(NUMBER_OF_JOINTS):
        joint_samples = all_samples[i::NUMBER_OF_JOINTS]
        phase = create_phase_vector(ALPHA_X, TAU, SAMPLE_RATE, num_of_samples)
        dmp = DMP(SAMPLE_RATE, TAU, ALPHA_Z, ALPHA_X, joint_samples, phase)
        joint_dmps.append(dmp)
        dmp.learn_weights()
        reproduced_movement = dmp.produce_movement(joint_samples[0], joint_samples[-1], TAU, phase)
        plt.subplot(ROWS, COLUMNS, i+1)
        plt.title(joint_names[i])
        plt.plot(time_space, joint_samples, label="Original Movement")
        plt.plot(time_space, reproduced_movement, label="Reproduced Movement")

    plt.legend()
    plt.show()