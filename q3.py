"""
q3.py — Repeated BACKWARD A* (Backward Replanning) with tie-breaking variants + Pygame visualization

Renders TWO views side-by-side:
- LEFT  : full (ground-truth) maze used for the run
- RIGHT : agent knowledge + search visualization

Controls:
- R : generate a new random maze and run again (max-g by default)
- 1 : run MAX-G on the current maze
- 2 : run MIN-G on the current maze
- ESC or close window : quit

Maze file loader (optional helper): readFile(fname) reads 0/1 tokens (space-separated), 1=blocked, 0=free.

Legend (colors):
GREY   = expanded / frontier / unknown (unseen)
PATH   = executed path (agent actually walked)
YELLOW = start + current agent position
BLUE   = goal
WHITE  = known free
BLACK  = known blocked
"""

from __future__ import annotations

import heapq
import argparse
import json
from typing import Callable, Dict, List, Optional, Tuple
from tqdm import tqdm
import time
import pygame
from constants import ROWS, START_NODE, END_NODE, BLACK, WHITE, GREY, YELLOW, BLUE, PATH, NODE_LENGTH, GRID_LENGTH, WINDOW_W, WINDOW_H, GAP
from custom_pq import CustomPQ_maxG, CustomPQ_minG
from q2 import repeated_forward_astar


# ---------------- FILE LOADER ----------------
def readMazes(fname: str) -> List[List[List[int]]]:
    """
    Reads a JSON file containing a list of mazes.
    Each maze is a list of ROWS lists, each with ROWS int values (0=free, 1=blocked).
    Returns a list of maze[r][c] grids.
    """
    with open(fname, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    mazes: List[List[List[int]]] = []
    for idx, grid in enumerate(data):
        if len(grid) != ROWS or any(len(row) != ROWS for row in grid):
            raise ValueError(f"Maze {idx}: expected {ROWS}x{ROWS}, got {len(grid)}x{len(grid[0]) if grid else 0}")
        maze = [[int(v) for v in row] for row in grid]
        maze[START_NODE[0]][START_NODE[1]] = 0
        maze[END_NODE[0]][END_NODE[1]] = 0
        mazes.append(maze)
    return mazes

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def get_neighbors(pos, grid):
    neighbors = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dx, dy in directions:
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]):
            neighbors.append((nx, ny))
    return neighbors

def a_star(grid, start, goal, callbacks=None):
    if grid[start[0]][start[1]] == 1 or grid[goal[0]][goal[1]] == 1:
        return None, 0
    open_list = CustomPQ_maxG()

    g = {start: 0} # g[(x, y)] = cost
    closed_list = set() 
    parent = {start: None} # parent cells
    open_list.push(manhattan_distance(start, goal), 0, start) # (f, tie, (x, y))

    expanded = 0

    while open_list:
        f, tie, curr_cell = open_list.pop()
        if curr_cell in closed_list:
            continue

        closed_list.add(curr_cell)
        expanded += 1

        if callbacks and "on_expand" in callbacks:
            callbacks["on_expand"](curr_cell)

        if curr_cell == goal:
            path = []
            while curr_cell is not None:
                path.append(curr_cell)
                curr_cell = parent[curr_cell]
            return path[::-1], expanded

        for neighbor in get_neighbors(curr_cell, grid):
            r, c = neighbor
            if grid[r][c] == 1:
                continue
            if neighbor in closed_list:
                continue

            new_g = 1 + g[curr_cell]
            if neighbor not in g or new_g < g[neighbor]:
                parent[neighbor] = curr_cell
                g[neighbor] = new_g
                f = new_g + manhattan_distance(neighbor, goal)
                open_list.push(f, new_g, neighbor)

    return None, expanded

def repeated_backward_astar(
    actual_maze: List[List[int]],
    start: Tuple[int, int] = START_NODE,
    goal: Tuple[int, int] = END_NODE,
    visualize_callbacks: Optional[Dict[str, Callable[[Tuple[int, int]], None]]] = None,
) -> Tuple[bool, List[Tuple[int, int]], int, int]:
    
    # TODO: Implement Backward A* with max_g tie-braking strategy.
    # Use heapq for standard priority queue implementation and name your max_g heap class as `CustomPQ_maxG` and use it. 
    n = len(actual_maze)
    if actual_maze[start[0]][start[1]] == 1 or actual_maze[goal[0]][goal[1]] == 1:
        return False, [start], 0, 0

    agent_grid = [[0 for _ in range(n)] for _ in range(n)]

    current = start
    final_path = [current]
    expanded_total = 0
    replans = 0

    cb = visualize_callbacks

    for neighbor in get_neighbors(current, actual_maze):
        r, c = neighbor
        if actual_maze[r][c] == 1:
            agent_grid[r][c] = 1
        if cb:
            cb["on_observe"](neighbor, is_blocked=(actual_maze[r][c] == 1))
        
    while current != goal:
        replans += 1
        if cb:
            cb["on_replan"]()

        path, expanded = a_star(agent_grid, goal, current, callbacks=cb) # backwards search
        expanded_total += expanded

        if not path:
            return False, final_path, expanded_total, replans
        
        path = path[::-1]
        for cell in path[1:]:
            r, c = cell
            if actual_maze[r][c] == 1:
                agent_grid[r][c] = 1
                if cb:
                    cb["on_observe"](cell, is_blocked=True)
                break

            current = cell
            final_path.append(current)
            if cb:
                cb["on_move"](current)

            for neighbor in get_neighbors(current, actual_maze):
                r, c = neighbor
                if actual_maze[r][c] == 1:
                    agent_grid[r][c] = 1
                if cb:
                    cb["on_observe"](neighbor, is_blocked=(actual_maze[r][c] == 1))
                

            if current == goal:
                return True, final_path, expanded_total, replans
            
    return True, final_path, expanded_total, replans

def show_astar_search(win: pygame.Surface, actual_maze: List[List[int]], algo: str, fps: int = 240, step_delay_ms: int = 0, save_path: Optional[str] = None) -> None:
    # [BONUS] TODO: Place your visualization code here.
    # This function should display the maze used, the agent's knowledge, and the search process as the agent plans and executes.
    # As a reference, this function takes pygame Surface 'win' to draw on, the actual maze grid, the algorithm name for labeling, 
    # and optional parameters for controlling the visualization speed and saving a screenshot.
    # You are free to use other visualization libraries other than pygame. 
    # You can call repeated_forward_astar with visualize_callbacks that update the Pygame display as the agent plans and executes.
    # In the end it should store the visualization as a PNG file if save_path is provided, or default to "vis_{algo}.png".
    # print(f"[{algo}] found={found}  executed_steps={len(executed)-1}  expanded={expanded}  replans={replans}")
    n = len(actual_maze)
    cell_size = NODE_LENGTH

    left_offset_x = 0
    right_offset_x = GRID_LENGTH + GAP

    knowledge = [[-1 for _ in range(n)] for _ in range(n)]  # -1=unknown, 0=free, 1=blocked
    expanded_cells = set()
    executed_path = []
    agent_pos = START_NODE

    clock = pygame.time.Clock()

    def draw_cell(x_offset, row, col, color):
        rect = pygame.Rect(
            x_offset + col * cell_size,
            row * cell_size,
            cell_size,
            cell_size,
        )
        pygame.draw.rect(win, color, rect)

    def draw_left_grid(): # actual maze
        for r in range(n):
            for c in range(n):
                if actual_maze[r][c] == 1:
                    draw_cell(left_offset_x, r, c, BLACK)
                else:
                    draw_cell(left_offset_x, r, c, WHITE)

        for cell in executed_path:
            draw_cell(left_offset_x, cell[0], cell[1], PATH)

        draw_cell(left_offset_x, START_NODE[0], START_NODE[1], YELLOW)
        draw_cell(left_offset_x, END_NODE[0], END_NODE[1], BLUE)
        draw_cell(left_offset_x, agent_pos[0], agent_pos[1], YELLOW)

    def draw_right_grid():
        """Agent knowledge + expanded cells overlay."""
        for r in range(n):
            for c in range(n):
                if knowledge[r][c] == 1:
                    draw_cell(right_offset_x, r, c, BLACK)
                elif knowledge[r][c] == 0:
                    draw_cell(right_offset_x, r, c, WHITE)
                else:
                    draw_cell(right_offset_x, r, c, GREY)

        # Expanded cells from current A* search
        for cell in expanded_cells:
            if cell != START_NODE and cell != END_NODE:
                draw_cell(right_offset_x, cell[0], cell[1], GREY)

        # Executed path
        for cell in executed_path:
            draw_cell(right_offset_x, cell[0], cell[1], PATH)

        # Start, goal, agent
        draw_cell(right_offset_x, START_NODE[0], START_NODE[1], YELLOW)
        draw_cell(right_offset_x, END_NODE[0], END_NODE[1], BLUE)
        draw_cell(right_offset_x, agent_pos[0], agent_pos[1], YELLOW)

    def refresh():
        win.fill(BLACK)
        draw_left_grid()
        draw_right_grid()
        pygame.display.flip()
        clock.tick(fps)
        if step_delay_ms > 0:
            pygame.time.delay(step_delay_ms)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                raise SystemExit
            
    # callbacks for visualization
    def on_expand(cell):
        expanded_cells.add(cell)
        # refresh()

    def on_move(cell):
        nonlocal agent_pos
        agent_pos = cell
        executed_path.append(cell)
        refresh()

    def on_observe(cell, is_blocked):
        r, c = cell
        knowledge[r][c] = 1 if is_blocked else 0

    def on_replan():
        expanded_cells.clear()
        refresh()

    visualize_callbacks = {
        "on_expand": on_expand,
        "on_move": on_move,
        "on_observe": on_observe,
        "on_replan": on_replan,
    }

    found, executed, expanded, replans = repeated_backward_astar(
        actual_maze,
        START_NODE,
        END_NODE,
        visualize_callbacks=visualize_callbacks,
    )

    print(f"[{algo}] found={found}  executed_steps={len(executed)-1}  expanded={expanded}  replans={replans}")

    refresh()
    
    if save_path is None:
        save_path = f"vis_{algo}.png"

    # If 'win' is the display surface (it is), this works:
    pygame.image.save(win, save_path)
    print(f"Saved the visualization -> {save_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Q3: Repeated Backward A*")
    parser.add_argument("--maze_file", type=str, required=True,
                        help="Path to input JSON file containing a list of mazes")
    parser.add_argument("--output", type=str, default="results_q3.json",
                        help="Path to output JSON results file")
    parser.add_argument("--show_vis", action="store_true",
                        help="[Bonus] If set, show Pygame visualization for the selected maze")
    parser.add_argument("--maze_vis_id", type=int, default=0,
                        help="[Bonus] maze_id (index) 0 ... 49 among 50 grid worlds")
    parser.add_argument("--save_vis_path", type=str, default="q3-vis-max-g.png",
                        help="[Bonus] If set, save visualization to this PNG file")
    args = parser.parse_args()

    mazes = readMazes(args.maze_file)
    results: List[Dict] = []

    for maze_id in tqdm(range(len(mazes)), desc="Processing mazes"):
        entry: Dict = {"maze_id": maze_id}

        t0 = time.perf_counter()
        found, executed, expanded, replans = repeated_backward_astar(
            actual_maze=mazes[maze_id],
            start=START_NODE,
            goal=END_NODE,
        )
        t1 = time.perf_counter()

        entry["bwd"] = {
            "found": found,
            "path_length": len(executed) - 1 if found else -1,
            "expanded": expanded,
            "replans": replans,
            "runtime_ms": (t1 - t0) * 1000,
        }

        t0 = time.perf_counter()
        found, executed, expanded, replans = repeated_forward_astar(
            actual_maze=mazes[maze_id],
            start=START_NODE,
            goal=END_NODE,
            tie_breaking="max_g",
        )
        t1 = time.perf_counter()

        entry["fwd"] = {
            "found": found,
            "path_length": len(executed) - 1 if found else -1,
            "expanded": expanded,
            "replans": replans,
            "runtime_ms": (t1 - t0) * 1000,
        }

        results.append(entry)

    if args.show_vis:
        # In case, PyGame is used for visualization, this code initializes a window and runs the visualization for the selected maze and algorithm.
        # Feel free to modify this code if you use a different visualization library or approach.
        pygame.init()
        win = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Repeated Backward A* Visualization")
        clock = pygame.time.Clock()
        selected_maze = mazes[args.maze_vis_id]
        current_algo = "max_g"
        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
        running = True
        while running:
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        current_algo = "max_g"
                        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
                    elif event.key == pygame.K_1:
                        current_algo = "max_g"
                        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
                    elif event.key == pygame.K_2:
                        current_algo = "min_g"
                        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
            pygame.display.flip()

        pygame.quit()

    with open(args.output, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"Results for {len(results)} mazes written to {args.output}")


if __name__ == "__main__":
    main()