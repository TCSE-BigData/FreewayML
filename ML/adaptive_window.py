import numpy as np

class AdaptiveWindow:
    def __init__(self, window_size, max_batches, threshold=0.5):
        self.window_size = window_size
        self.max_batches = max_batches
        self.batches = []
        self.distances = []
        self.threshold = threshold
        self.saved_features = None  # 用于存储特征的属性

    def add_batch(self, new_batch):
        if len(self.batches) >= self.max_batches:
            self.batches.pop(0)
        self.batches.append(new_batch)
        self.update_distances(new_batch)

    def calculate_distance(self, batch1, batch2):
        min_rows = min(batch1.shape[0], batch2.shape[0])
        return np.linalg.norm(batch1[:min_rows] - batch2[:min_rows])

    def update_distances(self, new_batch):
        current_distances = [self.calculate_distance(new_batch, batch) for batch in self.batches[:-1]]
        self.distances.append(current_distances)
        if current_distances:
            self.apply_decay()

    def apply_decay(self):
        if self.distances and any(self.distances):
            sorted_indices = np.argsort([np.mean(dist) if np.any(dist) else 0 for dist in self.distances])
            decay_rates = self.rank_to_decay_rates(sorted_indices)
            for i, rate in enumerate(decay_rates):
                if i < len(self.batches):
                    # Calculate the number of samples to keep
                    keep_count = int(len(self.batches[i]) * rate)
                    # Randomly select 'keep_count' samples
                    if keep_count < len(self.batches[i]):
                        self.batches[i] = self.batches[i][
                                          np.random.choice(len(self.batches[i]), keep_count, replace=False), :]

    def rank_to_decay_rates(self, sorted_indices):
        return [1.0 - (i / len(sorted_indices)) * 0.1 for i in sorted_indices]

    def should_update(self):
        if self.distances and any(self.distances):
            disorder = np.std([np.mean(dist) if np.any(dist) else 0 for dist in self.distances])
            return disorder > self.threshold
        return False

    def save_current_features(self):
        """保存当前窗口中的所有批次特征。"""
        self.saved_features = np.vstack([batch[:, :-1] for batch in self.batches])
        # 深复制每个批次的特征

    def clear_window(self):
        """清空当前窗口的批次和距离信息。"""
        self.batches = []
        self.distances = []

