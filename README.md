# Repeated and Adaptive A* in Gridworlds

This project implements and compares three variants of the A* search algorithm in partially observable 2D gridworld environments:

- Repeated Forward A*
- Repeated Backward A*
- Adaptive A*

The algorithms are evaluated on randomly generated 101x101 mazes. Performance metrics include:

- Total node expansions
- Number of replans
- Final path length
- Runtime (milliseconds)

The environment simulates partial observability where the agent does not know blocked cells in advance and must replan as obstacles are discovered.

---

## Generating Test Mazes

Use the maze generator:
```

python gen_test_json.py --num_mazes 50 --seed 42 --output mazes.json

```

Arguments:

| Argument      | Description |
|--------------|------------|
| --num_mazes  | Number of 101x101 mazes to generate |
| --seed       | Random seed for reproducibility |
| --output     | Output JSON file path |

Example:

```

python gen_test_json.py --num_mazes 100 --seed 123 --output test_mazes.json

```


This produces a JSON file containing a list of mazes.

---

## Running Experiments (Repeated Forward A*)

```

python main.py --maze_file mazes.json

```

Arguments:

| Argument | Description |
|----------|------------|
| --maze_file | Path to input JSON file (required) |
| --output | Output results JSON file (default: results_q2.json) |
| --tie_braking | max_g, min_g, or both (default: both) |
| --show_vis | Enable Pygame visualization |
| --maze_vis_id | Index of maze to visualize |
| --save_vis_path | Save visualization as PNG |

---

### Tie-Breaking Options

- `max_g` → Break ties in favor of larger g-values  
- `min_g` → Break ties in favor of smaller g-values  
- `both` → Run both variants

Example:

```

python main.py --maze_file mazes.json --tie_braking max_g

```

---

## Visualization Controls

If `--show_vis` is enabled:

- `1` → Run max_g
- `2` → Run min_g
- `R` → Rerun current algorithm
- `ESC` → Exit visualization window

Example:

```

python main.py --maze_file mazes.json --show_vis --maze_vis_id 3

```

To save the visualization:

```

python main.py --maze_file mazes.json --show_vis --save_vis_path output.png

```

---

## Output Format

Results are written to a JSON file with the following structure:

```
[
    {
        "maze_id": 0,
            "max_g": {
                "found": true,
                "path_length": 204,
                "expanded": 18532,
                "replans": 7,
                "runtime_ms": 82.4
            },
        "min_g": {
            ...
        }
    }
]
```



### Metrics

- `found` → Whether a path to the goal exists  
- `path_length` → Length of executed path  
- `expanded` → Total number of expanded nodes across replans  
- `replans` → Number of times A* was restarted  
- `runtime_ms` → Total runtime in milliseconds  

---
