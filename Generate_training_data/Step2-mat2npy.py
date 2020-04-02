import numpy as np
from scipy import io

#training
mat = io.loadmat('label.mat')
mat_t = np.transpose(mat['label'])
mat=mat_t.astype('float32')
np.save('label.npy', mat)

mat = io.loadmat('data_x.mat')
mat_t = np.transpose(mat['data_x'])
mat=mat_t.astype('float32')
np.save('data_x.npy', mat)

mat = io.loadmat('data_y.mat')
mat_t = np.transpose(mat['data_y'])
mat=mat_t.astype('float32')
np.save('data_y.npy', mat)

