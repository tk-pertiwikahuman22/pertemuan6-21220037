import streamlit as st
import pandas as pd
import numpy as np


# Implementasi sederhana SVM menggunakan kernel linear
class SimpleSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)  # Pastikan label hanya -1 atau 1
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (
                        2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx])
                    )
                    self.bias -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.weights) - self.bias
        return np.sign(approx)


# Fungsi untuk memproses data
def process_data(df, target_col):
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    y = np.where(y == df[target_col].unique()[0], 0, 1)  # Binerkan label
    return X, y


# Streamlit UI
st.title("SVM Classification Web App")
st.sidebar.title("Upload Dataset & Parameters")

uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df)

    target_col = st.sidebar.selectbox("Select Target Column", df.columns)
    test_size = st.sidebar.slider("Test Data Proportion", 0.1, 0.5, 0.2)
    learning_rate = st.sidebar.number_input("Learning Rate", 0.0001, 1.0, 0.001, step=0.0001)
    lambda_param = st.sidebar.number_input("Lambda (Regularization)", 0.01, 1.0, 0.1, step=0.01)
    n_iters = st.sidebar.slider("Number of Iterations", 100, 5000, 1000)

    if st.sidebar.button("Run SVM"):
        # Split data into train and test sets
        X, y = process_data(df, target_col)
        split_idx = int((1 - test_size) * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train SVM
        svm = SimpleSVM(learning_rate=learning_rate, lambda_param=lambda_param, n_iters=n_iters)
        svm.fit(X_train, y_train)

        # Evaluate model
        y_pred = svm.predict(X_test)
        accuracy = np.mean(y_pred == y_test)

        st.write("### Model Results:")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Predictions:", y_pred)
