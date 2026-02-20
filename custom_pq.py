import heapq

class CustomPQ_maxG:
    def __init__(self):
        self.heap = []
        self.counter = 0

    def push(self, f, g, node):
        heapq.heappush(self.heap, (f, -g, self.counter, node)) # negative g for max-heap
        self.counter += 1

    def pop(self):
        f, g, _, node = heapq.heappop(self.heap)
        return f, -g, node 

    def __len__(self):
        return len(self.heap)

    def __bool__(self):
        return len(self.heap) > 0


class CustomPQ_minG:
    def __init__(self):
        self.heap = []
        self.counter = 0

    def push(self, f, g, node):
        heapq.heappush(self.heap, (f, g, self.counter, node))
        self.counter += 1

    def pop(self):
        f, g, _, node = heapq.heappop(self.heap)
        return f, g, node

    def __len__(self):
        return len(self.heap)

    def __bool__(self):
        return len(self.heap) > 0