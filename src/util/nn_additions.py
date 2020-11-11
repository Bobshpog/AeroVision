import numpy as np
import torch
from pytorch_lightning.metrics import Metric
from torch.utils.data import Sampler


class MeanMetric(Metric):
    def __init__(self, foo, compute_on_step=False, dist_sync_on_step=False):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)
        self.add_state("value", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.foo = foo

    def update(self, y_hat: torch.Tensor, y: torch.Tensor) -> None:
        self.value += self.foo(y_hat, y).mean()
        self.count += 1

    def compute(self):
        return self.value / self.count


class HistMetric(Metric):
    def __init__(self, foo, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("hist", default=[], dist_reduce_fx="cat")
        self.foo = foo

    def update(self, y_hat: torch.Tensor, y: torch.Tensor):
        result = self.foo(y_hat, y).flatten()
        self.hist.append(result)

    def compute(self):
        return torch.cat(self.hist).cpu().numpy()


class SubsetChoiceSampler(Sampler):
    def __init__(self, subset_size, total_size):
        self.subset_size = subset_size
        self.total_range = range(total_size)

    def __iter__(self):
        return (self.total_range[i] for i in np.random.choice(self.total_range, size=self.subset_size, replace=False))

    def __len__(self):
        return self.subset_size