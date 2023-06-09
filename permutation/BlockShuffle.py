import numpy as np


def doScramble(X, key, flip, blockSize):
    input_shape = X.shape
    assert (X.dtype == np.uint8)
    assert (input_shape[1] % blockSize[0] == 0)
    assert (input_shape[2] % blockSize[1] == 0)
    assert (input_shape[3] == blockSize[2])
    numBlock = np.int32([input_shape[1] / blockSize[0], input_shape[2] / blockSize[1]])
    numCh = blockSize[2]
    X = np.reshape(X, (input_shape[0], numBlock[0], blockSize[0], numBlock[1], blockSize[1], numCh))
    X = np.transpose(X, (0, 1, 3, 2, 4, 5))
    X = np.reshape(X, (input_shape[0], numBlock[0], numBlock[1], blockSize[0] * blockSize[1] * numCh))

    X0 = X & 0xF  # lower 4 bits
    X1 = X >> 4  # upper 4 bits
    X = np.concatenate((X0, X1), axis=3)

    X[..., flip] = (15 - X[..., flip].astype(np.int32)).astype(np.uint8)  # random bit flip
    X = X[..., key]
    X[..., flip] = (15 - X[..., flip].astype(np.int32)).astype(np.uint8)

    #  recombine
    d = blockSize[0] * blockSize[1] * numCh
    X0 = X[..., :d]
    X1 = X[..., d:]
    X = (X1 << 4) + X0

    X = np.reshape(X, (input_shape[0], numBlock[0], numBlock[1], blockSize[0], blockSize[1], numCh))
    X = np.transpose(X, (0, 1, 3, 2, 4, 5))
    X = np.reshape(X, (input_shape[0], numBlock[0] * blockSize[0], numBlock[1] * blockSize[1], numCh))
    return X


class BlockScramble:
    def __init__(self, blockSize, seed):
        self.blockSize = blockSize
        key = self.genKey(seed)
        self.key = key
        self.rev = (key > key.size / 2)
        self.invKey = np.argsort(key)

    def genKey(self, seed):
        np.random.seed(seed)
        key = self.blockSize[0] * self.blockSize[1] * self.blockSize[2]
        key = np.arange(key * 2, dtype=np.uint32)
        np.random.shuffle(key)
        np.random.seed()
        return key

    def Scramble(self, X):
        XX = (X * 255).astype(np.uint8)
        XX = doScramble(XX, self.key, self.rev, self.blockSize)
        return XX.astype('float32') / 255.0

    def Decramble(self, X):
        XX = (X * 255).astype(np.uint8)
        XX = doScramble(XX, self.invKey, self.rev, self.blockSize)
        return XX.astype('float32') / 255.0
