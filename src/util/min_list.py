from dataclasses import dataclass


@dataclass
class MinCounter:
    max_count: int

    def __post_init__(self):
        if self.max_count<0:
            raise ValueError
        self.data = [float("inf")] * self.max_count
        self.count = 0

    def add(self, x: float) -> bool:
        """
        Args:
            x:number to add to min list
        Returns:
            weather we didn't update the list for max_count epochs
        """
        if x < self.data[0]:
            self.count = 0
            self.data[-1] = x
            self.data.sort()
        else:
            self.count += 1
        return self.count >= self.max_count
