import random
import json
import tkinter as tk
GRID_SIZE = 10
P_BLOCKED = 0.4
CELL_SIZE = 30

seed = 100
random.seed(seed)
def generate_grid(n, p_blocked=0.3):
    """ Generate n x n grid"""
    grid = [[0 for _ in range(n)] for _ in range(n)]
    unvisited = [(i, j) for i in range(n) for j in range(n)]
    unvisited = set(unvisited)
    blocked = set()

    def dfs(x, y):
        if (x, y) not in unvisited or (x, y) in blocked:
            return
        unvisited.remove((x, y))
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and (nx, ny) in unvisited:
                if random.random() < p_blocked:
                    blocked.add((nx, ny))
                    unvisited.remove((nx, ny))
                    grid[nx][ny] = 1
                    return
                dfs(nx, ny)
    
    while unvisited:
        x, y = random.choice(list(unvisited))
        dfs(x, y)

    return grid

def save_grid_json(grid, path):
    data = {
        "rows": len(grid),
        "cols": len(grid[0]),
        "grid": grid,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def load_grid_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["grid"]

def draw_grid(canvas, grid, path=None, search=None, start=None, goal=None):
    canvas.delete("all")
    rows, cols = len(grid), len(grid[0])
    path_set = set(path) if path else set()
    search_set = set(search) if search else set()
    for r in range(rows):
        for c in range(cols):
            x1, y1 = c * CELL_SIZE, r * CELL_SIZE
            x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
            if (r, c) == start:
                color = "green"
            elif (r, c) == goal:
                color = "red"
            elif (r, c) in path_set:
                color = "blue"
            elif(r, c) in search_set:
                color = "purple"
            elif grid[r][c] == 1:
                color = "black"
            else:
                color = "white"
            canvas.create_rectangle(x1, y1, x2, y2, fill=color)

def show_grid(grid, path=None, search=None, start=None, goal=None):
    rows, cols = len(grid), len(grid[0])
    root = tk.Tk()
    root.title("Maze")

    canvas = tk.Canvas(root, width=cols * CELL_SIZE, height=rows * CELL_SIZE)
    canvas.pack()

    draw_grid(canvas, grid, path, search, start, goal)
    root.mainloop()

def main(): 
    grid = generate_grid(GRID_SIZE, P_BLOCKED)
    show_grid(grid)

if __name__ == "__main__":
    main()