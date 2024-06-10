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
        x = np.array(x)
        y = np.array(y)
        if self.theta == None:
            self.theta = np.zeros(x.shape[1])
        n = x.shape[0]

        for j in range(self.max_iter):
            prev_theta = self.theta.copy()
            grad_l = np.zeros(x.shape[1])
            H = np.zeros((x.shape[1], x.shape[1]))
            
            for i in range(n):
                xi = x[i, :].reshape((-1, 1))
                g  = 1/(1 + np.exp(-np.dot(self.theta.T, xi)))
                grad_l = grad_l + (y[i] - g) * xi
                H = H + g * (1 - g) * np.dot(xi, xi.T)

            grad_l /= -n
            H /= n
            
            self.theta = self.theta - np.dot(np.linalg.inv(H), grad_l)
            if np.linalg.norm(self.theta - prev_theta) < self.eps:
                break
            
            


        # newton : theta = theta - f/f'
        # g  = 1/(1+np.exp(np.dot(theta.T,x[i,:])))
        # f  = (-1/n)*sum((y[i] - g)*x[i,:])
        # f' = (1/n)*sum(g*(1-g)*x[i,:]*x[i,:].T)
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        x = np.array(x)
        return 1/(1 + np.exp(-np.dot(self.theta.T, x)))

        # *** END CODE HERE ***