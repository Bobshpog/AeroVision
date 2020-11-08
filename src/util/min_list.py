from dataclasses import dataclass


@dataclass
class MinCounter:
    max_count: int

    def __post_init__(self):
        if self.max_count < 0:
            raise ValueError
        self.data = [-self.max_count + 1, float("inf")]
        self.count = 0

    def add(self, x: float, epoch: int, ) -> bool:
        """
        Args:
            x:number to add to min list
        Returns:
            weather we didn't update the list for max_count epochs
        """
        self.data = [x for x in self.data if ((epoch - x[0]) <= self.max_count)]
        if (not bool(self.data)):
            # if list is empty
            return True
        self.data.append((epoch, x))
        self.data.sort(key=lambda x: x[1])
        return False
