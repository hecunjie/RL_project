import numpy as np
import dataset


class LinUCB:
    """
    LinUCB algorithm implementation
    """

    def __init__(self, alpha, context="user"):
        """
        Parameters
        ----------
        alpha : number
            LinUCB parameter
        context: string
            'user' or 'both'(item+user): what to use as a feature vector
        """
        self.n_features = len(dataset.features[0])
        if context == "user":
            self.context = 1
        elif context == "both":
            self.context = 2
            self.n_features *= 2

        self.A = np.array([np.identity(self.n_features)] * dataset.n_arms)
        self.A_inv = np.array([np.identity(self.n_features)] * dataset.n_arms)
        self.b = np.zeros((dataset.n_arms, self.n_features, 1))
        self.alpha = round(alpha, 1)
        self.algorithm = "LinUCB(alpha=" + str(self.alpha) + ",context:" + context + ")"

    def reinit(self,nums_training_articles):
        self.A = np.vstack([self.A, np.array([np.identity(self.n_features)] * nums_training_articles)])
        self.A_inv =  np.vstack([self.A_inv, np.array([np.identity(self.n_features)] * nums_training_articles)])
        self.b = np.vstack([self.b,np.zeros((nums_training_articles, self.n_features, 1))])

    def choose_arm(self, t, user, pool_idx):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        A_inv = self.A_inv[pool_idx]
        b = self.b[pool_idx]

        n_pool = len(pool_idx)

        user = np.array([user] * n_pool)
        if self.context == 1:
            x = user
        else:
            x = np.hstack((user, dataset.features[pool_idx]))

        x = x.reshape(n_pool, self.n_features, 1)

        theta = A_inv @ b

        p = np.transpose(theta, (0, 2, 1)) @ x + self.alpha * np.sqrt(
            np.transpose(x, (0, 2, 1)) @ A_inv @ x
        )
        return np.argmax(p)

    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        a = pool_idx[displayed]  # displayed article's index
        if self.context == 1:
            x = np.array(user)
        else:
            x = np.hstack((user, dataset.features[a]))

        x = x.reshape((self.n_features, 1))

        self.A[a] += x @ x.T
        self.b[a] += reward * x
        self.A_inv[a] = np.linalg.inv(self.A[a])



class KernelUCB:
    """
    KernelUCB algorithm implementation with Gaussian Kernel (RBF)
    """

    def __init__(self, alpha, context="user", gamma=0.1):
        """
        Parameters
        ----------
        alpha : float
            UCB parameter (exploration factor)
        context : str
            'user' or 'both' (item+user): what to use as a feature vector
        gamma : float
            RBF kernel width parameter (controls similarity decay)
        max_events : int
            Maximum number of events to keep in memory (used for limiting the size of similarity matrix)
        """
        self.alpha = round(alpha, 1)
        self.gamma = gamma
        self.context = context
        self.n_features = len(dataset.features[0])

        if context == "user":
            self.context = 1
        elif context == "both":
            self.context = 2
            self.n_features *= 2

        # Initialize parameters for each arm (article)
        self.Kernles = [np.identity(1)] * dataset.n_arms
        self.K_inv = [np.identity(1)] * dataset.n_arms
        self.max_events = [[] for _ in range(dataset.n_arms)]
        self.b = [np.zeros(1) for _ in range(dataset.n_arms)]
        self.algorithm = "KernelUCB(alpha=" + str(self.alpha) + ",context:" + context + ")"

    def calculate_rbf(self, x, X, gamma=0.1):
        """
        Compute the Gaussian (RBF) kernel between x and each row in X.
        """
        return np.exp(-gamma * np.linalg.norm(x - X, axis=1) ** 2)

    def calculate_similarity_matrix(self, A, events, event):
        """
        Compute the similarity matrix based on RBF kernel between event and existing events.
        """
        n_events = len(events)
        events = np.array(events)
        if n_events == 0:
            return np.identity(1)
        elif n_events < 100:
            B = np.ones((n_events + 1, n_events + 1))
            B[:-1, :-1] = A.copy()
            new_rbf = self.calculate_rbf(event, events)
            B[-1, :-1] = new_rbf
            B[:-1, -1] = new_rbf
        else:
            B = np.ones((100, 100))
            B[:n_events - 1, :n_events - 1] = A[1:, 1:].copy()
            new_rbf = self.calculate_rbf(event, events)
            B[-1, :-1] = new_rbf[1:]
            B[:-1, -1] = new_rbf[1:]

        return B

    def choose_arm(self, t, user, pool_idx):
        """
        Choose the best arm from the pool based on the current context and UCB rule.
        Parameters
        ----------
        t : int
            Number of trials
        user : array
            User features
        pool_idx : array of indexes
            Pool indexes for article identification
        is_kernel : bool
            Whether to use kernelized features
        """
        b = [self.b[i] for i in pool_idx]

        n_pool = len(pool_idx)

        # Depending on context, use user features or both (user + item)
        if self.context == 1:
            x = user
        else:
            x = np.hstack([user, dataset.features[pool_idx]])
        
        max_ucb,maxind = -np.inf,0
        for i in range(n_pool):
            if len(self.max_events[pool_idx[i]]) > 0:
                # print(x,np.array(self.max_events[pool_idx[i]]))
                K_star = self.calculate_rbf(x, np.array(self.max_events[pool_idx[i]]))
                # print(K_star.reshape(1,-1).shape,self.K_inv[pool_idx[i]].shape,b[i].shape)
                mean = K_star.reshape(1,-1) @ self.K_inv[pool_idx[i]] @ b[i]
                std = np.sqrt(1 - K_star @ self.K_inv[pool_idx[i]] @ K_star)
                if mean + self.alpha * std > max_ucb:
                    max_ucb = mean + self.alpha * std
            else:
                max_ucb = 0
                maxind = i

        return pool_idx[maxind]

    def update(self, displayed, reward, user, pool_idx):
        """
        Updates the parameters (A, b) based on the displayed article and the received reward.
        Parameters
        ----------
        displayed : int
            Displayed article index relative to the pool
        reward : binary
            Whether the user clicked or not (1 for click, 0 for no click)
        user : array
            User features
        pool_idx : array of indexes
            Pool indexes for article identification
        """
        a = pool_idx[displayed]  # displayed article's index
        if self.context == 1:
            x = np.array(user)
        else:
            x = np.hstack([user, dataset.features[a]])

        history = len(self.max_events[a])

        if history == 0 :
            self.b[a] = np.array([reward]).reshape((1,1))
            self.max_events[a].append(x)
        else:
            self.Kernles[a] = self.calculate_similarity_matrix(self.Kernles[a], self.max_events[a], x)
            self.K_inv[a] = np.linalg.inv(self.Kernles[a]+np.eye(self.Kernles[a].shape[0]))
            if len(self.max_events[a]) < 100:
                self.max_events[a].append(x)
                self.b[a] = np.vstack([self.b[a], reward])
            else:
                self.max_events[a] = self.max_events[a][1:] + [x]
                self.b[a] = np.vstack([self.b[a][1:], reward])

class ThompsonSampling:
    """
    Thompson sampling algorithm implementation
    """

    def __init__(self):
        self.algorithm = "TS"
        self.alpha = np.ones(dataset.n_arms)
        self.beta = np.ones(dataset.n_arms)

    def choose_arm(self, t, user, pool_idx):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        theta = np.random.beta(self.alpha[pool_idx], self.beta[pool_idx])
        return np.argmax(theta)

    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        a = pool_idx[displayed]

        self.alpha[a] += reward
        self.beta[a] += 1 - reward


class Ucb1:
    """
    UCB 1 algorithm implementation
    """

    def __init__(self, alpha):
        """
        Parameters
        ----------
        alpha : number
            ucb parameter
        """

        self.alpha = round(alpha, 1)
        self.algorithm = "UCB1 (alpha=" + str(self.alpha) + ")"

        self.q = np.zeros(dataset.n_arms)  # average reward for each arm
        self.n = np.ones(dataset.n_arms)  # number of times each arm was chosen

    def choose_arm(self, t, user, pool_idx):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        ucbs = self.q[pool_idx] + np.sqrt(self.alpha * np.log(t + 1) / self.n[pool_idx])
        return np.argmax(ucbs)

    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        a = pool_idx[displayed]

        self.n[a] += 1
        self.q[a] += (reward - self.q[a]) / self.n[a]


class Egreedy:
    """
    Epsilon greedy algorithm implementation
    """

    def __init__(self, epsilon):
        """
        Parameters
        ----------
        epsilon : number
            Egreedy parameter
        """

        self.e = round(epsilon, 1)  # epsilon parameter for Egreedy
        self.algorithm = "Egreedy (epsilon=" + str(self.e) + ")"
        self.q = np.zeros(dataset.n_arms)  # average reward for each arm
        self.n = np.zeros(dataset.n_arms)  # number of times each arm was chosen

    def choose_arm(self, t, user, pool_idx):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        p = np.random.rand()
        if p > self.e:
            return np.argmax(self.q[pool_idx])
        else:
            return np.random.randint(low=0, high=len(pool_idx))

    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        a = pool_idx[displayed]

        self.n[a] += 1
        self.q[a] += (reward - self.q[a]) / self.n[a]
