import numpy as np
import struct
import gzip

def load_images(filename):
    with gzip.open(filename, 'rb') if filename.endswith('.gz') else open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        X = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
        return X / 255.0  # normalize to [0,1], shape (N,784)

def load_labels(filename):
    with gzip.open(filename, 'rb') if filename.endswith('.gz') else open(filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        y = np.frombuffer(f.read(), dtype=np.uint8)
        return y  # shape (N,)

def one_hot(labels, num_classes=10):
    # Turn [3,0,4] into rows of the identity matrix
    return np.eye(num_classes)[labels]  # shape (N,10)

def softmax(z):
    # z: (N,10)
    z = z - np.max(z, axis=1, keepdims=True)  # stability
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)  # (N,10)

def cross_entropy(preds, Y):
    # preds, Y: (N,10)
    # Average negative log-prob of the true class
    return -np.mean(np.sum(Y * np.log(preds + 1e-9), axis=1))

def predict(X, W, b):
    # logits: (N,10) = (N,784)@(784,10) + (1,10)
    logits = X @ W + b
    return softmax(logits)  # (N,10)

def train(X, Y, lr=0.1, epochs=10):
    N, D = X.shape     # N samples, D=784 features
    C = Y.shape[1]     # C=10 classes
    W = np.random.randn(D, C) * 0.01
    b = np.zeros((1, C))

    for epoch in range(epochs):
        logits = X @ W + b          # (N,10)
        P = softmax(logits)         # (N,10)
        loss = cross_entropy(P, Y)  # scalar

        # Gradients
        dlogits = (P - Y) / N       # (N,10)
        dW = X.T @ dlogits          # (784,10)
        db = np.sum(dlogits, axis=0, keepdims=True)  # (1,10)

        # Update
        W -= lr * dW
        b -= lr * db

        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

    return W, b

def accuracy(X, y_true, W, b):
    preds = np.argmax(predict(X, W, b), axis=1)  # (N,)
    return np.mean(preds == y_true)

def main():
    print("Loading data...")
    X_train = load_images("train-images-idx3-ubyte")[:5000]
    y_train_raw = load_labels("train-labels-idx1-ubyte")[:5000]
    Y_train = one_hot(y_train_raw, 10)

    X_test = load_images("t10k-images-idx3-ubyte")[:500]
    y_test_raw = load_labels("t10k-labels-idx1-ubyte")[:500]

    print("Training model...")
    W, b = train(X_train, Y_train, lr=0.5, epochs=10)

    print("Evaluating...")
    acc = accuracy(X_test, y_test_raw, W, b)
    print(f"Test Accuracy: {acc*100:.2f}%")

    # Single prediction demo
    idx = 3
    p = predict(X_test[idx:idx+1], W, b)  # shape (1,10)
    print("Probs:", p.ravel())
    print(f"Predicted: {np.argmax(p)}, Actual: {y_test_raw[idx]}")

main()