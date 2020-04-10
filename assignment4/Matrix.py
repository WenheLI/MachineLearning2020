class Matrix:
    def __init__(self, data, w, h):
        super().__init__()
        if len(data) != w * h:
            raise ArithmeticError("Mismatch dimension")
        self.w = w
        self.h = h
        self.data = [[0 for i in range(w)] for i in range(h)]
        for idx in range(len(data)):
            self.data[idx - h*(idx//h)][idx//h] = data[idx]
    def dot(self, m):
        temp = 0
        for x in range(len(self.w)):
            for y in range(len(self.h)):
                temp += self.data[x][y] * m[x][y]
        