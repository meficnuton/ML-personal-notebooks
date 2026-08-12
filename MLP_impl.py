import sys
import numpy as np

class NeuralNetMLP:
    """Feedforward neural network / Multi-layer perceptron classifier."""
    def __init__(self, n_hidden: int = 30, l2: float = 0.0, epochs: int = 100,
                eta: float = 0.001, shuffle: bool = True, mini_batch_size: int = 1,
                seed: int | None = None):
        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.mini_batch_size = mini_batch_size
        self.eval_ = {'cost' : [], 'train_acc' : [], 'valid_acc' : []}

    def _onehot(self, y: np.ndarray, n_classes: int) -> np.ndarray:
        """Encode labels into one-hot representation."""
        onehot = np.zeros(n_classes, y.shape[0])
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.0
            return onehot.T
        
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Compute logistic function (sigmoid)."""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))
    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute forward propagation step"""
        # Step 1 & 2: Hidden layer
        z_h = X @ self.w_h + self.b_h
        a_h = self._sigmoid(z_h)

        # Step 3 & 4: Output layer
        z_out = a_h @ self.w_out + self.h_out
        a_out = self._sigmoid(z_out)

        return z_h, a_h, z_out, a_out
        
    def _compute_cost(self, y_enc: np.ndarray, output: np.ndarray) -> float:
        """Compute cost function"""
        L2_term = self.l2 * (np.sum(self.w_h ** 2.0) + np.sum(self.w_out ** 2.0))
        term1 = -y_enc * np.log(output + 1e-9) # added epsilon to prevent log(0)
        term2 = (1.0 - y_enc) * np.log(1.0 - output + 1e-9)
        cost = np.sum(term1 - term2) + L2_term
        return float(cost)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        _, _, _, a_out = self._forward(X)
        return np.argmax(a_out, axis = 1)
    def fit(self, X_train: np.ndarray,
                  y_train: np.ndarray,
                  X_valid: np.ndarray,
                  y_valid: np.ndarray):
        """Learn weights from training data"""
        n_output = np.unique(y_train).shape[0]
        n_features = X_train.shape[1]
        # Weight initialization
        self.b_h = np.zeros(self.n_hidden)
        self.w_h = self.random.normal(loc=0.0, scale=0.1, size=(n_features))
        y_train_enc = self._onehot(y_train, n_output)

        for i in range(self.epochs):
            indices = np.arange(X_train.shape[0])
            if self.shuffle:
                self.random.shuffle(indices)
            for start_idx in range(0, indices.shape[0] - self.mini_batch_size + 1, self.mini_batch_size):
                batch_idx = indices[start_idx:start_idx + self.mini_batch_size]
                # Forward propagation
                _, a_h, _, a_out = self._forward(X_train[batch_idx])
                
                # Backpropagation
                delta_out = a_out - y_train_enc[batch_idx]
                sigmoid_derivative_h = a_h * (1.0 - a_h)
                
                delta_h = (delta_out @ self.w_out.T) * sigmoid_derivative_h
                
                grad_w_h = X_train[batch_idx].T @ delta_h
                grad_b_h = np.sum(delta_h, axis=0)
                
                grad_w_out = a_h.T @ delta_out
                grad_b_out = np.sum(delta_out, axis=0)
                
                # Regularization and weight updates
                self.w_h -= self.eta * (grad_w_h + self.l2 * self.w_h)
                self.b_h -= self.eta * grad_b_h
                
                self.w_out -= self.eta * (grad_w_out + self.l2 * self.w_out)
                self.b_out -= self.eta * grad_b_out

            # Evaluation
            _, _, _, a_out = self._forward(X_train)
            cost = self._compute_cost(y_enc=y_train_enc, output=a_out)
            
            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)
            
            # Replaced deprecated np.float with built-in float
            train_acc = float(np.sum(y_train == y_train_pred)) / X_train.shape[0]
            valid_acc = float(np.sum(y_valid == y_valid_pred)) / X_valid.shape[0]
            
            # Modern f-string formatting
            sys.stderr.write(f'\r{i+1:0{len(str(self.epochs))}d}/{self.epochs} | '
                             f'Cost: {cost:.2f} | '
                             f'Train/Valid Acc.: {train_acc*100:.2f}%/{valid_acc*100:.2f}%')
            sys.stderr.flush()
            
            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)
            
        return self