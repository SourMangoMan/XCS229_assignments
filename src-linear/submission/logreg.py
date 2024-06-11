import numpy as np

class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        def l1_norm(a,b = None):
            if b == None:
                b = np.zeros(len(a))

            sum = 0
            for i in range(len(a)):
                sum += np.abs(a[i] - b[i])

            return sum

        n = x.shape[0]

        x = np.array(x)
        y = np.array(y)

        self.theta, grad_l = np.zeros((x.shape[1], 1))
        
        H = np.zeros((x.shape[1], x.shape[1]))
        
        for i in range(self.max_iter):
            prev_theta = (self.theta).copy()
            for i in range(n):
                x_i = (x[i,:]).reshape(-1,1)

                g = 1/(1 + np.exp(self.theta.T @ x_i))
                grad_l += (g - y[i])*x_i
                H += g*(1 - g)*(x_i @ x_i.T)

            grad_l /= n
            H /= n

            self.theta -= np.linalg.inv(H) @ grad_l

            if l1_norm(self.theta, prev_theta) < self.eps:
                break

        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        # *** END CODE HERE ***