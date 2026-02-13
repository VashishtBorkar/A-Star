import heapq
def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def get_neighbors(pos, grid):
    neighbors = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dx, dy in directions:
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 0:
            neighbors.append((nx, ny))
    
    return neighbors

def a_star(grid, start, goal):
    open_list = [] # heap
    heapq.heappush(open_list, (manhattan_distance(start, goal), start)) # (f, (x, y))
    closed_list = set() 
    g = {start: 0} # g[(x, y)] = cost
    parent = {start: None} # parent cells

    while open_list:
        f, curr_cell = heapq.heappop(open_list)

        if curr_cell == goal:
            path = []
            while curr_cell is not None:
                path.append(curr_cell)
                curr_cell = parent[curr_cell]
            return path[::-1]

        closed_list.add(curr_cell)

        for neighbor in get_neighbors(curr_cell, grid):
            if neighbor in closed_list:
                continue

            new_g = 1 + g[curr_cell]

            if neighbor not in g or new_g < g[neighbor]:
                parent[neighbor] = curr_cell
                g[neighbor] = new_g
                f = new_g + manhattan_distance(neighbor, goal)
                heapq.heappush(open_list, (f, neighbor))

def repeated_a_star(true_grid, start, goal, forward=True):
    n = len(true_grid)
    agent_grid = [[0 for _ in range(n)] for _ in range(n)]
    current = start
    final_path = [current]

    while current != goal:
        if forward:
            path = a_star(agent_grid, current, goal)
        else:
            path = a_star(agent_grid, goal, current)

        if not path:
            return None
        
        if not forward:
            path = path[::-1]
        
        for cell in path[1:]:
            r, c = cell
            if true_grid[r][c] == 1:
                agent_grid[r][c] = 1
                break

            current = cell
            final_path.append(current)

            if current == goal:
                return final_path
            
    return final_path
        
def adaptive_a_star(grid, start, goal):
    pass
