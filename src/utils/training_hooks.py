"""
Shared Detectron2 training hooks.

Provides EarlyStoppingHook used by multiple Mask R-CNN training scripts.
"""
from detectron2.engine import HookBase


class EarlyStoppingHook(HookBase):
    """
    Early Stopping Hook for Detectron2.
    Stops training if the monitored metric does not improve for `patience` evaluations.
    """
    def __init__(self, eval_period, patience=10, metric_name="segm/AP", mode="max"):
        self.eval_period = eval_period
        self.patience = patience
        self.metric_name = metric_name
        self.mode = mode
        self.best_metric = float('-inf') if mode == "max" else float('inf')
        self.counter = 0
        self.should_stop = False

    def after_step(self):
        if (self.trainer.iter + 1) % self.eval_period == 0:
            storage = self.trainer.storage
            try:
                latest = storage.latest()
                if self.metric_name in latest:
                    current_metric = latest[self.metric_name][0]

                    improved = (self.mode == "max" and current_metric > self.best_metric) or \
                              (self.mode == "min" and current_metric < self.best_metric)

                    if improved:
                        self.best_metric = current_metric
                        self.counter = 0
                        print(f"\nNew best {self.metric_name}={current_metric:.4f}")
                    else:
                        self.counter += 1
                        print(f"\nNo improvement ({self.counter}/{self.patience}). Best: {self.best_metric:.4f}")

                    if self.counter >= self.patience:
                        print(f"\nEARLY STOPPING: No improvement for {self.patience} evaluations.")
                        print(f"  Best {self.metric_name}={self.best_metric:.4f}")
                        self.should_stop = True
                        self.trainer.iter = self.trainer.max_iter - 1
            except (KeyError, AttributeError):
                pass
