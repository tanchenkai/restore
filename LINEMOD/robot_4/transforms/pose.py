import os
import numpy as np
for ind in range(1258):
    pose_path = os.path.join('{}.npy'.format(ind))
    pose = np.load(pose_path)
    R = pose[0:3,0:3]
    T = pose[0:3,3]
    rotfile = os.path.join('data','rot{}.rot'.format(ind))
    trafile = os.path.join('data','tra{}.tra'.format(ind))
    np.savetxt(rotfile,R,header="3 3",comments="")
    np.savetxt(trafile,T,header="1 3",comments="")
