import numpy as np
import torch
from pytorch_lightning.metrics import Metric
from torch.utils.data import Sampler


class ReduceMetric(Metric):
    def __init__(self, foo, compute_on_step=False, dist_sync_on_step=False):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)
        if isinstance(foo,tuple):
            self.reduction=foo[-1]
        else:
            self.reduction='mean'
        if self.reduction=='mean':
            self.add_state("value", default=torch.tensor(0,dtype=torch.float,device='cuda'), dist_reduce_fx="sum")
            self.add_state("count", default=torch.tensor(0,dtype=torch.float,device='cuda'), dist_reduce_fx="sum")
        elif self.reduction=='max':
            self.add_state("value", default=torch.tensor(0, dtype=torch.float, device='cuda'), dist_reduce_fx=torch.max)
        else:
            raise NotImplementedError("only mean and max are supported")
        if isinstance(foo,tuple) and len(foo)==3:
            self.foo,self.max_val=foo[:2]
        else:
            self.foo = foo
            self.max_val=1

    def update(self, y_hat: torch.Tensor, y: torch.Tensor) -> None:
        if self.reduction =='mean':
            self.value += self.foo(y_hat, y).mean()
            self.count += 1
        elif self.reuction =='max':
            self.value= torch.max(self.value,self.foo(y_hat,y))

    def compute(self):
        if self.reduction=='mean':
            return self.value / (self.max_val*self.count)
        elif self.reduction=='max':
            return self.value/self.max_val


class HistMetric(Metric):
    def __init__(self, foo, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("hist", default=[], dist_reduce_fx="cat")
        if isinstance(foo,tuple):
            self.foo,self.max_val=foo
        else:
            self.foo = foo
            self.max_val=1

    def update(self, y_hat: torch.Tensor, y: torch.Tensor):
        result = self.foo(y_hat, y).flatten()
        self.hist.append(result)

    def compute(self):
        return torch.cat(self.hist).cpu().numpy()/self.max_val


class SubsetChoiceSampler(Sampler):
    def __init__(self, subset_size, total_size):
        self.subset_size = subset_size
        self.total_range = range(total_size)

    def __iter__(self):
        return (self.total_range[i] for i in np.random.choice(self.total_range, size=self.subset_size, replace=False))

    def __len__(self):
        return self.subset_size