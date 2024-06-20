import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tsplib95
import numpy as np
import random
import time


def load_tsp_problem(file_path):
    with open(file_path, 'r') as f:
        problem = tsplib95.parse(f.read())

    nodes = list(problem.get_nodes())
    print(f"Nodes: {nodes}")

    if problem.node_coords:
        points = np.array([problem.node_coords[node] for node in nodes])
    else:
        print("No node coordinates found.")

    return points


def calculate_total_distance(points, order, distance_matrix):
    # Obliczenie całkowitego dystansu trasy dla danej kolejności węzłów
    total_distance = 0
    num_points = len(points)
    for i in range(num_points):
        total_distance += distance_matrix[order[i]][order[(i + 1) % num_points]]
    return total_distance


def create_distance_matrix(points):
    # Stworzenie macierzy odległości między punktami na podstawie ich współrzędnych
    num_points = len(points)
    distance_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            distance_matrix[i][j] = np.linalg.norm(points[i] - points[j])
    return distance_matrix


def swap_operator(order):
    # Operator zamiany dwóch losowych węzłów w kolejności
    new_order = order.copy()
    index1, index2 = random.sample(range(len(order)), 2)
    new_order[index1], new_order[index2] = new_order[index2], new_order[index1]
    return new_order


def reverse_segment_operator(order):
    # Operator odwrócenia losowego segmentu w kolejności
    new_order = order.copy()
    index1, index2 = random.sample(range(len(order)), 2)
    if index1 > index2:
        index1, index2 = index2, index1
    new_order[index1:index2 + 1] = reversed(new_order[index1:index2 + 1])
    return new_order


def insert_operator(order):
    # Operator wstawienia losowego węzła w losowe miejsce w kolejności
    new_order = order.copy()
    index1, index2 = random.sample(range(len(order)), 2)
    city = new_order.pop(index1)
    new_order.insert(index2, city)
    return new_order


def slide_operator(order):
    # Operator przesunięcia losowego węzła na inne losowe miejsce w kolejności
    new_order = order.copy()
    index1, index2 = random.sample(range(len(order)), 2)
    if index1 < index2:
        city = new_order.pop(index1)
        new_order.insert(index2, city)
    else:
        city = new_order.pop(index1)
        new_order.insert(index2 + 1, city)
    return new_order


def tsp_heuristic(points, distance_matrix, operators, max_iterations=1000000, update_interval=10000, initial_temp=1000, cooling_rate=0.9999):
    # Heurystyczne rozwiązanie problemu komiwojażera
    num_points = len(points)
    order = list(range(num_points))
    random.shuffle(order)

    best_order = order
    best_distance = calculate_total_distance(points, order, distance_matrix)
    current_order = order
    current_distance = best_distance

    distances = [best_distance]
    temp = initial_temp
    no_improvement_count = 0
    max_no_improvement = 10000

    for i in range(max_iterations):
        new_order = random.choice(operators)(current_order)
        new_distance = calculate_total_distance(points, new_order, distance_matrix)

        if new_distance < current_distance or random.random() < np.exp((current_distance - new_distance) / temp):
            current_order = new_order
            current_distance = new_distance
            no_improvement_count = 0
            if current_distance < best_distance:
                best_order = current_order
                best_distance = current_distance
                distances.append(new_distance)
        else:
            no_improvement_count += 1

        temp *= cooling_rate

        if i % update_interval == 0 or i == max_iterations - 1:
            yield points[best_order], distances, best_distance

        if no_improvement_count >= max_no_improvement:
            print(f"No improvement for {max_no_improvement} iterations. Stopping early.")
            break

    yield points[best_order], distances, best_distance


def visualize_tsp_solution(points, distance_matrix, operators, max_iterations=1000000, update_interval=10000, output_file="tsp_solution.gif"):
    # Wizualizacja rozwiązania problemu komiwojażera za pomocą animacji
    fig, ax = plt.subplots()
    ax.set_title('TSP Heuristic Solution')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    sc = ax.scatter(points[:, 0], points[:, 1], c='b', marker='o', s=100, edgecolors='k')
    line, = ax.plot([], [], 'r--', lw=2)
    text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        positions, distances, current_distance = frame
        line.set_data(positions[:, 0], positions[:, 1])
        text.set_text(f'Distance: {current_distance:.2f}')
        return sc, line, text

    def generate_frames(points, distance_matrix, operators, max_iterations, update_interval):
        for positions, distances, current_distance in tsp_heuristic(points, distance_matrix, operators, max_iterations, update_interval):
            yield positions, distances, current_distance

    ani = animation.FuncAnimation(fig, update,
                                  frames=generate_frames(points, distance_matrix, operators, max_iterations, update_interval),
                                  init_func=init, blit=True, cache_frame_data=False)

    # Zapisanie animacji
    ani.save(output_file, writer='pillow', fps=5)
    plt.show()

file_path = 'att532.tsp'
chosen_points = load_tsp_problem(file_path)
distance_matrix = create_distance_matrix(chosen_points)
chosen_operators = [swap_operator, reverse_segment_operator, insert_operator, slide_operator]

start_time = time.time()
# Generowanie gifa i zapisanie ostatecznego wyniku
final_state = None
for state in tsp_heuristic(chosen_points, distance_matrix, chosen_operators, max_iterations=1000000, update_interval=10000):
    final_state = state

visualize_tsp_solution(chosen_points, distance_matrix, chosen_operators, max_iterations=1000000, update_interval=10000)
end_time = time.time()

final_positions, final_distances, final_distance = final_state
optimal_distance = 27686

percentage_difference = (final_distance - optimal_distance) / optimal_distance * 100

print(f"Final distance: {final_distance}")
print(f"Percentage difference from optimal: {percentage_difference:.2f}%")
print(f"Execution time: {end_time - start_time:.2f} seconds")
