from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from time import sleep
import numpy as np

reachy = ReachySDK(host="127.0.0.1")
# put the joints in stiff mode
reachy.turn_on('r_arm')

A = np.array([
  [0, 0, -0.9, 0.3],
  [0, 0.9, 0, -0.4],  
  [0.9, 0, 0, -0.3],
  [0, 0, 0, 0.9],  
])

joint_pos_A = reachy.r_arm.inverse_kinematics(A)

# use the goto function
goto({joint: pos for joint,pos in zip(reachy.r_arm.joints.values(), joint_pos_A)}, duration=3.0)
sleep(5.0)

# put the joints back to compliant mode
# use turn_off_smoothly to prevent the arm from falling hard
reachy.turn_off_smoothly('r_arm')