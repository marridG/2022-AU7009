import os
import numpy as np

fn = "1.jgp"
print(list(os.path.splitext(fn)).insert(1, "12312312"))

a = np.arange(12)
print(a.reshape(-1, 1))
print(np.row_stack((np.ones_like(a), a)).T)

b = np.array([[False, False], [True, False]])
print(b[0, 1] == 0, b[0, 1] == 1, b[1, 0] == 0, b[1, 0] == 1, )
