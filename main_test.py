import numpy as np
import pandas as pd

sim_client_state = np.load('simulation/sim_client_feature.npy')
i = 0
for s in sim_client_state:
    print("idx: %2d, poi: %d, %d, %d, real: %.3f, %.3f, cur: %.3f, %.3f, pred: %.3f, %.3f, size: %.3f" % (i, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9]))
    i += 1