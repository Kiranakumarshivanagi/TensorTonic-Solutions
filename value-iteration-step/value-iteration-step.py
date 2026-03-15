import numpy as np

def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """

    V = np.array(values, dtype=float)
    T = np.array(transitions, dtype=float)
    R = np.array(rewards, dtype=float)

    # Expected future values
    future = np.sum(T * V, axis=2)

    # Q-values
    Q = R + gamma * future

    # Take best action
    V_new = np.max(Q, axis=1)

    return V_new.tolist()