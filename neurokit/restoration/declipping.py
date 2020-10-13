import numpy as np
import scipy.signal as ss


def create_frames(signal, frame_size=256, step_size=None, window=None):
    if step_size is None:
        step_size = frame_size // 4

    offsets = np.arange(0, len(signal) - frame_size + 1, step_size)
    frames = np.zeros((len(offsets), frame_size))
    for n, offset in enumerate(offsets):
        frames[n] = signal[offset:offset + frame_size]

    if window is not None:
        frames *= ss.get_window(window, frame_size)

    return frames


def create_signal(frames, step_size, window='boxcar'):
    num_frames, frame_size = frames.shape
    signal_size = step_size * (num_frames - 1) + frame_size

    signal = np.zeros(signal_size)
    overlap = np.zeros(signal_size)

    window_values = ss.get_window(window, frame_size)

    offsets = step_size * np.arange(num_frames)
    for n, offset in enumerate(offsets):
        signal[offset:offset + frame_size] += frames[n]
        overlap[offset:offset + frame_size] += window_values

    return signal / overlap


def dct_dictionary(frame_size, num_atoms):
    window = ss.hann(frame_size)

    t = np.arange(frame_size)  # time
    k = np.arange(num_atoms)  # frequency

    D = np.cos(np.pi / num_atoms
               * np.dot((t[np.newaxis, :].T + 0.5), k[np.newaxis, :] + 0.5))
    D = np.diag(window).dot(D)

    norm = np.sqrt(np.sum(D**2, 0))

    return D / norm


def hard_threshold(A, k):
    A_abs = np.abs(A)
    return A * (A_abs >= -np.partition(-A_abs, k - 1, axis=0)[k - 1])


def consistent_dictionary_learning(frames, M, k, n1, n2, D=None, A=None,
                                   num_iterations=100):
    if D is None:
        D = dct_dictionary(frames.shape[1], 2 * frames.shape[1])
    if A is None:
        A = np.zeros((D.shape[1], frames.shape[0]))

    Y = frames.T
    M_pos = M.T > 0
    M_neg = M.T < 0

    for iteration in range(num_iterations):
        # Sparse coding
        μ1 = 1 / np.linalg.norm(D, 2)**2

        for i in range(n1):
            R = Y - D @ A
            R[M_pos] = np.maximum(R[M_pos], 0)
            R[M_neg] = np.minimum(R[M_neg], 0)

            A = A + μ1 * D.T @ R  # gradient descent
            A = hard_threshold(A, k)

        # Dictionary learning
        empty_atoms = np.where(np.sum(A**2, 1) == 0)[0]
        D = np.delete(D, empty_atoms, axis=1)
        A = np.delete(A, empty_atoms, axis=0)

        μ2 = 1 / np.linalg.norm(A, 2)**2

        for j in range(n2):
            R = Y - D @ A
            R[M_pos] = np.maximum(R[M_pos], 0)
            R[M_neg] = np.minimum(R[M_neg], 0)

            D = D + μ2 * R @ A.T  # gradient descent
            D = D / np.sqrt(np.sum(D**2, 0))  # normalize

    return D, A


def declip_signal(signal, low, high, frame_size=256, step_size=64):
    signal_len = len(signal)

    if signal_len % frame_size != 0:
        signal = np.pad(signal, (0, frame_size - signal_len % frame_size))

    frames = create_frames(signal, frame_size, step_size, window='hamming')

    clipping_mask = 1 * (signal >= high) - 1 * (signal <= low)
    M = create_frames(clipping_mask, frame_size, step_size)

    D, A = consistent_dictionary_learning(frames, M, 32, 20, 20,
                                          num_iterations=50)

    reconstructed_frames = (D @ A).T
    reconstructed = create_signal(reconstructed_frames, step_size,
                                  window='hamming')

    declipped = signal.copy()
    declipped[clipping_mask != 0] == reconstructed[clipping_mask != 0]

    return reconstructed[:signal_len]
