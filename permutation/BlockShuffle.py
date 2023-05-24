import pickle

import numpy as np


class BlockScramble:
    def __init__(self, blockSize_filename, seed):
        if isinstance(blockSize_filename, str):
            self.load(blockSize_filename)
        else:
            self.blockSize = blockSize_filename
            key = self.genKey(seed)
            self.setKey(key)

    def setKey(self, key):
        self.key = key
        self.rev = (key > key.size / 2)
        self.invKey = np.argsort(key)

    def load(self, filename):
        with open(filename, 'rb') as fin:
            self.blockSize, self.key = pickle.load(fin)
        self.setKey(self.key)

    def save(self, filename):  # pkl
        with open(filename, 'wb') as fout:
            pickle.dump([self.blockSize, self.key], fout)

    def genKey(self, seed):
        np.random.seed(seed)
        key = self.blockSize[0] * self.blockSize[1] * self.blockSize[2]
        key = np.arange(key * 2, dtype=np.uint32)
        np.random.shuffle(key)
        np.random.seed()
        return key

    def Scramble(self, X):
        XX = (X * 255).astype(np.uint8)
        XX = self.doScramble(XX, self.key, self.rev)
        return XX.astype('float32') / 255.0

    def Decramble(self, X):
        XX = (X * 255).astype(np.uint8)
        XX = self.doScramble(XX, self.invKey, self.rev)
        return XX.astype('float32') / 255.0

    def doScramble(self, X, ord, rev):
        s = X.shape
        assert (X.dtype == np.uint8)
        assert (s[1] % self.blockSize[0] == 0)
        assert (s[2] % self.blockSize[1] == 0)
        assert (s[3] == self.blockSize[2])
        numBlock = np.int32([s[1] / self.blockSize[0], s[2] / self.blockSize[1]])
        numCh = self.blockSize[2]
        X = np.reshape(X, (s[0], numBlock[0], self.blockSize[0], numBlock[1], self.blockSize[1], numCh))
        X = np.transpose(X, (0, 1, 3, 2, 4, 5))
        X = np.reshape(X, (s[0], numBlock[0], numBlock[1], self.blockSize[0] * self.blockSize[1] * numCh))

        X0 = X & 0xF  # lower 4 bits
        X1 = X >> 4  # upper 4 bits
        X = np.concatenate((X0, X1), axis=3)

        X[..., rev] = (15 - X[..., rev].astype(np.int32)).astype(np.uint8)  # random bit flip
        X = X[..., ord]
        X[..., rev] = (15 - X[..., rev].astype(np.int32)).astype(np.uint8)

        #  recombine
        d = self.blockSize[0] * self.blockSize[1] * numCh
        X0 = X[..., :d]
        X1 = X[..., d:]
        X = (X1 << 4) + X0

        X = np.reshape(X, (s[0], numBlock[0], numBlock[1], self.blockSize[0], self.blockSize[1], numCh))
        X = np.transpose(X, (0, 1, 3, 2, 4, 5))
        X = np.reshape(X, (s[0], numBlock[0] * self.blockSize[0], numBlock[1] * self.blockSize[1], numCh))
        return X
