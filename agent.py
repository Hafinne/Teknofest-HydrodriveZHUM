# agent.py
 
import numpy as np
import pickle
 
 
def _to_index(state):
    """
    State ne gelirse gelsin (tuple, numpy array, list)
    güvenli şekilde (int, int, int) tuple'ına çevirir.
    """
    if isinstance(state, np.ndarray):
        return tuple(int(x) for x in state)
    elif isinstance(state, (tuple, list)):
        return tuple(int(x) for x in state)
    else:
        return (int(state),)
 
 
class QLearningAgent:
    """
    Q-Öğrenme ajanı.
    State uzayı: (T_idx x SoC_idx x Tamb_idx) = 8 x 5 x 6
    Eylem uzayı: 5
    """
 
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995):
        self.alpha          = alpha
        self.gamma          = gamma
        self.epsilon        = epsilon
        self.epsilon_min    = epsilon_min
        self.epsilon_decay  = epsilon_decay
 
        # Q tablosu: 8 (T) x 5 (SoC) x 6 (T_amb) x 5 (eylem)
        self.Q = np.zeros((8, 5, 6, 5))
 
    def select_action(self, state):
        """Epsilon-greedy eylem seçimi"""
        idx = _to_index(state)
        if np.random.random() < self.epsilon:
            return np.random.randint(5)
        return int(np.argmax(self.Q[idx]))
 
    def update(self, state, action, reward, next_state, done):
        """Bellman denklemiyle Q tablosunu güncelle"""
        idx      = _to_index(state)
        next_idx = _to_index(next_state)
 
        current_q = self.Q[idx][action]
 
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_idx])
 
        self.Q[idx][action] += self.alpha * (target - current_q)
 
    def decay_epsilon(self):
        """Her episode sonunda epsilon azalt"""
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)
 
    def save(self, path="q_table.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.Q, f)
        print(f"Q tablosu kaydedildi: {path}")
 
    def load(self, path="q_table.pkl"):
        with open(path, "rb") as f:
            self.Q = pickle.load(f)
        print(f"Q tablosu yüklendi: {path}")