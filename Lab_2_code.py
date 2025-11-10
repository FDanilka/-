import math
import numpy as np
import matplotlib.pyplot as plt
import time


class Main_class:

    def __init__(self, func: str, x_start: float, x_end: float, eps=0.01) -> None:
        self.func_str = func
        self.x_start = x_start
        self.x_end = x_end
        self.eps = eps

        self.func = self._create_func()

        self.L = self._get_L()
        self.points = [(self.x_start, self.func(self.x_start)), (self.x_end, self.func(self.x_end))]
        self.intersect_cache = {}
        self.iteration_count = 0

    def step(self) -> bool:
        self.iteration_count += 1

        x_new, lower_bound = self._find_next_point()

        f_new = self.func(x_new)

        self.points.append((x_new, f_new))
        self.points.sort(key=lambda p: p[0])

        current_best_f = min(p[1] for p in self.points)
        gap = current_best_f - lower_bound

        return gap < self.eps

    def plot(self, show_solution=True) -> None:
        x_smooth = np.linspace(self.x_start, self.x_end, 1000)
        y_smooth = [self.func(x) for x in x_smooth]

        if show_solution:
            x_points = [p[0] for p in self.points]
            y_points = [p[1] for p in self.points]

            x_polyline, y_polyline = self._build_polyline()

        plt.figure(figsize=(12, 7))
        plt.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='Функция f(x)')

        if show_solution:
            plt.plot(x_polyline, y_polyline, 'r-', linewidth=1, label='Ломаная (нижняя оценка)')
            plt.scatter(x_points, y_points, color='red', s=20, label='Вычисленные точки')

            best_point = min(self.points, key=lambda p: p[1])
            plt.scatter([best_point[0]], [best_point[1]], color='green', s=100,
                        label=f'Минимум: f({best_point[0]:.3f}) = {best_point[1]:.3f}')

        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.title(f'Метод Пиявского - Итерация #{self.iteration_count}', fontsize=14)
        plt.grid(True, alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def solve(self, max_iterations=10000) -> tuple[float, float, float, int]:
        start_time = time.time()

        for _ in range(max_iterations):
            if self.step():
                break

        elapsed_time = time.time() - start_time
        min_point = min(self.points, key=lambda p: p[1])

        return min_point[0], min_point[1], elapsed_time, self.iteration_count

    def _find_next_point(self) -> tuple[float, float]:
        best_x, best_lower_bound = None, float('inf')
        best_interval_x1, best_interval_x2 = None, None

        for i in range(len(self.points) - 1):
            x1, f1 = self.points[i]
            x2, f2 = self.points[i + 1]
            x_intersect, y_intersect = self._get_intersect_coord(x1, f1, x2, f2)

            if (x1, x2) not in self.intersect_cache:
                self.intersect_cache[(x1, x2)] = (x_intersect, y_intersect)

            if y_intersect < best_lower_bound:
                best_lower_bound = y_intersect
                best_x = x_intersect
                best_interval_x1, best_interval_x2 = x1, x2

        del self.intersect_cache[(best_interval_x1, best_interval_x2)]
        return best_x, best_lower_bound

    def _get_intersect_coord(self, x1: float, f1: float, x2: float, f2: float) -> tuple[float, float]:
        if (x1, x2) in self.intersect_cache:
            return self.intersect_cache[(x1, x2)]

        x_intersect = ((f1 - f2) / (2 * self.L)) + ((x1 + x2) / 2)
        y_intersect = f1 - self.L * (x_intersect - x1)
        return x_intersect, y_intersect

    def _create_func(self):
        func_expr = self.func_str.replace("f(x)=", "").strip()
        return lambda x: eval(func_expr, {"math": math, "x": x})

    def _get_L(self) -> float:
        n_points = 1000
        x_samples = np.linspace(self.x_start, self.x_end, n_points)
        max_deriv = 0
        h = x_samples[1] - x_samples[0]

        for x in x_samples:
            if x - h >= self.x_start and x + h <= self.x_end:
                derivative = (self.func(x + h) - self.func(x - h)) / (2 * h)
                max_deriv = max(max_deriv, abs(derivative))

        return max_deriv * 1.2

    def _build_polyline(self):
        x_polyline, y_polyline = [], []

        for i in range(len(self.points) - 1):
            x1, f1 = self.points[i]
            x2, f2 = self.points[i + 1]

            x_polyline.append(x1)
            y_polyline.append(f1)

            x_intersect, y_intersect = self._get_intersect_coord(x1, f1, x2, f2)
            x_polyline.append(x_intersect)
            y_polyline.append(y_intersect)

        x_polyline.append(self.points[-1][0])
        y_polyline.append(self.points[-1][1])

        return x_polyline, y_polyline
