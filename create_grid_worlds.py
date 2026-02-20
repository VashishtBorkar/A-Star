"""
gen_test_json.py — Generate N random 101x101 mazes and save as mazes.json. Uses same algorithm as maze_generator.py.

Usage:
    python gen_test_json.py [--num_mazes N] [--seed S] [--output FILE]
"""
import json
import random
import argparse
import random
from constants import ROWS, P_BLOCKED
from tqdm import tqdm
import argparse

# set random seed for reproducibility
random.seed(42)

def create_maze() -> list:
    # TODO: Implement this function to generate and return a random maze as a 2D list of 0s and 1s.
    n = ROWS
    p_blocked = P_BLOCKED
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

def main():
    parser = argparse.ArgumentParser(description="Generate random mazes as JSON")
    parser.add_argument("--num_mazes", type=int, default=50,
                        help="Number of mazes to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="mazes.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    random.seed(args.seed)
    
    mazes = []
    for _ in tqdm(range(args.num_mazes), desc="Generating mazes"):  
        mazes.append(create_maze())

    with open(args.output, "w") as fp:
        json.dump(mazes, fp)
    print(f"Generated {args.num_mazes} mazes (seed={args.seed}) -> {args.output}")

if __name__ == "__main__":
    main()
