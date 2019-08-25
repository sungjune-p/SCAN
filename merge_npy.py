import numpy as np

A = np.load('./out/image_embeddings_1.npy')
print("Shape of A is ", A.shape)
B = np.load('./out/image_embeddings_2.npy')
print("Shape of B is ", B.shape)
C = np.concatenate((A, B), axis=0)
print("Shape of C merged with A and B is ", C.shape)
np.save('./out/image_emb.npy', C)
