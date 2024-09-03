import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = [np.random.randn(y, x) * 0.01 for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.zeros((y, 1)) for y in layers[1:]]

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def backpropagation(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # Feedforward
        activation = x
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        # Backward pass
        delta = (activations[-1] - y) * self.sigmoid_derivative(zs[-1])
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        nabla_b[-1] = delta

        for l in range(2, len(self.layers)):
            delta = np.dot(self.weights[-l+1].transpose(), delta) * self.sigmoid_derivative(zs[-l])
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            nabla_b[-l] = delta

        return nabla_w, nabla_b

    def update_mini_batch(self, mini_batch, learning_rate):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backpropagation(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

        self.weights = [w - (learning_rate / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def train(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        n = len(training_data)
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if test_data:
                n_test = len(test_data)
                accuracy = self.evaluate(test_data) / n_test
                print(f"Epoch {epoch + 1}: {accuracy * 100:.2f}% accuracy on test data")
            else:
                print(f"Epoch {epoch + 1} complete")

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

def load_mnist():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X.reshape((X.shape[0], -1, 1))
    X = X / 255.0
    y = y.astype(int).reshape(-1, 1)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y.reshape(-1)]

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_mnist()
    y_train_encoded = one_hot_encode(y_train, 10)
    y_test_encoded = one_hot_encode(y_test, 10)

    training_data = list(zip(X_train, y_train_encoded))
    test_data = list(zip(X_test, y_test.reshape(-1)))

    nn = NeuralNetwork([784, 128, 64, 10])
    nn.train(training_data, epochs=10, mini_batch_size=32, learning_rate=0.1, test_data=test_data)

    # Visualize some predictions
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        img = X_test[i].reshape(28, 28)
        ax.imshow(img, cmap='gray')
        prediction = np.argmax(nn.feedforward(X_test[i]))
        ax.set_title(f"Pred: {prediction}, True: {y_test[i][0]}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()
