import numpy as np


def cnn_forward(A, W, b, pad, stride):
    # m: num of training samples
    # h: height
    # w: width
    # c: channel (RGB)
    # n: num of filters
    # k: filter size
    (m, h, w, c) = A.shape
    (k, _, _, n) = W.shape

    # Compute new activiation map size
    h_new = int(np.floor((h - k + 2 * pad) / stride)) + 1
    w_new = int(np.floor((w - k + 2 * pad) / stride)) + 1

    # Create n activiation maps for m samples
    Z = np.zeros((m, h_new, w_new, n))

    # Create padding around A based on pad value
    A_pad = np.pad(A, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')

    # Fill up activation maps
    for i in range(m):
        for h_i in range(h_new):
            for w_i in range(w_new):
                for n_i in range(n):
                    h_start = h_i * stride
                    h_end = h_start + k
                    w_start = w_i * stride
                    w_end = w_start + k

                    A_slice = A_pad[i, h_start:h_end, w_start:w_end, :]
                    Z[i, h_i, w_i, n_i] = np.sum(np.multiply(A_slice, W[:, :, :, c])) + b[:, :, :, c]

    return Z


def main():
    # Filter (5,5,3 Activation Maps: 16)
    A = np.random.rand(8, 28, 28, 3)
    W = np.random.rand(5, 5, 3, 16)
    b = np.random.rand(1, 1, 1, 16)
    pad = 0
    stride = 1
    Z = cnn_forward(A, W, b, pad, stride)
    # Shape should be m * h_new * w_new * n
    print(Z.shape)


main()
