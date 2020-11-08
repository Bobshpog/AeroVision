from dataclasses import dataclass


@dataclass
class MinCounter:
    max_count: int

    def __post_init__(self):
        if self.max_count < 0:
            raise ValueError
        self.data = [(float("inf"), 0)]
        self.count = 0

    def add(self, epoch: int, x: float) -> bool:
        """
        Args:
            x:number to add to min list
        Returns:
            weather we didn't update the list for max_count epochs
        """
        self.data = [x for x in self.data if ((epoch - x[1]) > self.max_count)]
        if (not bool(self.data)):
            #if list is empty
            return True
        self.data.append((x, epoch))
        self.data.sort(key=lambda x: x[0])
        return False
