from grids import generate_grid, show_grid
from search import a_star, repeated_a_star, adaptive_a_star

GRID_SIZE = 20
P_BLOCKED = 0.4

def main():
    grid = generate_grid(GRID_SIZE, P_BLOCKED)
    # show_grid(grid)
    start = (0, 4)
    goal = (8, 8)
    optimal = a_star(grid, start, goal)
    forward = repeated_a_star(grid, start, goal, forward=True)
    backward = repeated_a_star(grid, start, goal, forward=False)
    print("Normal A*:\t", optimal)
    print("Forward A*:\t", forward)
    print("Backward A*:\t", backward)

    show_grid(grid, optimal, forward, start, goal)
    # show_grid(grid, optimal, start, goal)



    
if __name__ == "__main__":
    main()