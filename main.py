import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tsplib95
import numpy as np
import random


def load_tsp_problem(file_path):
    with open(file_path, 'r') as f:
        problem = tsplib95.parse(f.read())

    nodes = list(problem.get_nodes())
    print(f"Nodes: {nodes}")

    if problem.node_coords:
        points = np.array([problem.node_coords[node] for node in nodes])
    else:
        print("No node coordinates found. Generating random coordinates...")
        points = np.array([[random.random(), random.random()] for _ in nodes])

    return points


# Obliczanie całkowitej długości trasy
def calculate_total_distance(points, order):
    total_distance = 0
    num_points = len(points)
    for i in range(num_points):
        from_point = points[order[i]]
        to_point = points[order[(i + 1) % num_points]]
        total_distance += np.linalg.norm(from_point - to_point)
    return total_distance

# Operatory zmiany trasy
def swap_operator(order):
    new_order = order.copy()
    index1, index2 = random.sample(range(len(order)), 2)
    new_order[index1], new_order[index2] = new_order[index2], new_order[index1]
    return new_order

def reverse_segment_operator(order):
    new_order = order.copy()
    index1, index2 = random.sample(range(len(order)), 2)
    if index1 > index2:
        index1, index2 = index2, index1
    new_order[index1:index2 + 1] = reversed(new_order[index1:index2 + 1])
    return new_order

def insert_operator(order):
    new_order = order.copy()
    index1, index2 = random.sample(range(len(order)), 2)
    city = new_order.pop(index1)
    new_order.insert(index2, city)
    return new_order

def slide_operator(order):
    new_order = order.copy()
    index1, index2 = random.sample(range(len(order)), 2)
    if index1 < index2:
        city = new_order.pop(index1)
        new_order.insert(index2, city)
    else:
        city = new_order.pop(index1)
        new_order.insert(index2 + 1, city)
    return new_order

# Algorytm heurystyczny do rozwiązania TSP
def tsp_heuristic(points, operators, max_iterations=1000):
    num_points = len(points)
    order = list(range(num_points))
    random.shuffle(order)

    distances = [calculate_total_distance(points, order)]
    for i in range(max_iterations):
        new_order = random.choice(operators)(order)
        current_distance = calculate_total_distance(points, order)
        new_distance = calculate_total_distance(points, new_order)

        if new_distance < current_distance:
            order = new_order
            distances.append(new_distance)

        yield points[order], distances

# Wizualizacja algorytmu krok po kroku
def visualize_tsp_solution(points, operators, max_iterations=1000):
    fig, ax = plt.subplots()
    ax.set_title('TSP Heuristic Solution')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Inicjalizacja punktów
    sc = ax.scatter(points[:, 0], points[:, 1], c='b', marker='o', s=100, edgecolors='k')
    line, = ax.plot([], [], 'r--', lw=2)
    text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def update(frame):
        positions, distances = frame
        sc.set_offsets(positions)
        line.set_data(positions[:, 0], positions[:, 1])
        text.set_text(f'Dystans: {distances[-1]:.2f}')
        return sc, line, text

    iterations = tsp_heuristic(points, operators, max_iterations)
    ani = animation.FuncAnimation(fig, update, frames=iterations, blit=True)
    plt.show()

# Wybór problemu do rozwiązania i operatora
file_path = 'gr48.tsp'
chosen_points = load_tsp_problem(file_path)
chosen_operators = [swap_operator, reverse_segment_operator, insert_operator, slide_operator]

# Wizualizacja rozwiązania
visualize_tsp_solution(chosen_points, chosen_operators, max_iterations=1000)
