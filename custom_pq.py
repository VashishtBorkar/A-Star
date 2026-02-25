class CustomPQ_maxG: 
    def __init__(self):
        self.heap = []
        self.counter = 0
    
    def push(self, f, g, node):
        self.heap.append((f, -g, self.counter, node))
        self.counter += 1
        self._shift_up(len(self.heap) - 1)
    
    def pop(self):
        if not self.heap:
            raise IndexError("pop from empty heap")
        top = self.heap[0]
        last = self.heap.pop()
        if self.heap:
            self.heap[0] = last
            self._shift_down(0)
        return top[0], -top[1], top[3]

    def _shift_up(self, index):
        while index > 0:
            parent = (index - 1) // 2
            if self.heap[index] < self.heap[parent]:  
                self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
                index = parent
            else:
                break
    
    def _shift_down(self, index):
        size = len(self.heap)
        while True:
            left = 2 * index + 1
            right = 2 * index + 2
            smallest = index

            if left < size and self.heap[left] < self.heap[smallest]:
                smallest = left
            if right < size and self.heap[right] < self.heap[smallest]:
                smallest = right
            
            if smallest != index:
                self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
                index = smallest
            else:
                break

    def __len__(self):
        return len(self.heap)
    
    def __bool__(self):
        return len(self.heap) > 0


class CustomPQ_minG:
    def __init__(self):
        self.heap = []
        self.counter = 0

    def push(self, f, g, node):
        self.heap.append((f, g, self.counter, node))
        self.counter += 1
        self._shift_up(len(self.heap) - 1)
    
    def pop(self):
        if not self.heap:
            raise IndexError("pop from empty heap")
        top = self.heap[0]
        last = self.heap.pop()
        if self.heap:
            self.heap[0] = last
            self._shift_down(0)
        return top[0], top[1], top[3]

    def _shift_up(self, index):
        while index > 0:
            parent = (index - 1) // 2
            if self.heap[index] < self.heap[parent]: 
                self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
                index = parent
            else:
                break

    def _shift_down(self, index):
        size = len(self.heap)
        while True:
            left = 2 * index + 1
            right = 2 * index + 2
            smallest = index
            
            if left < size and self.heap[left] < self.heap[smallest]:  
                smallest = left
            if right < size and self.heap[right] < self.heap[smallest]:  
                smallest = right
            
            if smallest != index:
                self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
                index = smallest
            else:
                break

    def __len__(self):
        return len(self.heap)
    
    def __bool__(self):
        return len(self.heap) > 0