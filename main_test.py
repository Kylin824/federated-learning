import numpy as np
import pandas as pd

chosen_list = np.loadtxt('./simulation/valid_list_random.txt')
a = chosen_list[0]
idx = a[np.where(a != -1)]
print(idx)