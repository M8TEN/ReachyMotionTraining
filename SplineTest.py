import pickle
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

SHOW_POINTS: bool = False

names = [
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_arm_yaw",
    "l_elbow_pitch",
    "l_forearm_yaw",
    "l_wrist_pitch",
    "l_wrist_roll",
    "l_gripper"
]

# Your given 'knots' (the data points the curve must pass through)
# Example: 4 points in 2D (x, y coordinates)
# The points Q_i are provided as separate lists of coordinates for splprep
with open("PitchSamples.pkl", "rb") as file:
    sample_data = pickle.load(file)

for i in range(len(names)):
    y_data = sample_data[i::19]
    x_data = list(range(len(y_data))) #np.linspace(0, len(y_data)/60.0, len(y_data))
    points = [x_data, y_data]

    # 1. Calculate the B-spline representation (tck = knots, control points, degree)
    # k=3 specifies a cubic spline (degree 3) for C2 continuity
    # s=0 ensures interpolation (curve passes exactly through all points)
    tck, u = splprep(points, k=3, s=1)

    # tck[0] contains the calculated knot vector (t)
    # tck[1] contains the calculated control points (c)
    # tck[2] is the degree (k)

    # To explicitly get the control points:
    control_points = np.array(tck[1]).T

    # 2. Evaluate the spline at a finer set of parameter values for plotting
    u_new = np.linspace(u.min(), u.max(), 10000)
    x_new, y_new = splev(u_new, tck)

    # x_new and y_new contain the coordinates of the C2-continuous B-spline curve.

    plt.plot(x_new, y_new, label=names[i])
    if SHOW_POINTS:
        plt.plot(x_data, y_data, "o", label="Original Points")

plt.xlabel("Time")
plt.ylabel("Joint angle (deg)")
plt.legend()
plt.show()