import tensorflow as tf
import numpy as np

D1=np.load("data_y.npy")
D1_x = np.transpose(D1, (3,0,1,2))
print(D1_x.shape)
np.save('data_y.npy',D1_x)

D1=np.load("data_x.npy")
D1_x = np.transpose(D1, (3,0,1,2))
print(D1_x.shape)
np.save('data_x.npy',D1_x)

D1=np.load("label.npy")
D1_x = np.transpose(D1, (3,0,1,2))
print(D1_x.shape)
np.save('label.npy',D1_x)


