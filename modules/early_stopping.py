"""
Early Stopping implementation for training optimization
"""

class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.
    
    Args:
        patience: Number of epochs to wait after last improvement
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' for loss (lower is better) or 'max' for accuracy (higher is better)
    """
    def __init__(self, patience=7, min_delta=0.0, mode='min'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.min_delta = min_delta
        self.mode = mode
        
    def __call__(self, score):
        """
        Check if training should stop early.
        
        Args:
            score: Current validation score (loss or accuracy)
            
        Returns:
            bool: True if training should stop early
        """
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == 'min' and score < self.best_score - self.min_delta) or \
             (self.mode == 'max' and score > self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
