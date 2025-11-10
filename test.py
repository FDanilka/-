from Lab_2_code import Main_class

piyavsky = Main_class("f(x)=10 + x**2 - 10 * math.cos(2 * math.pi * x)", -5, 5, eps=0.01)
#Выполним 3 итераций
for _ in range(4):
    piyavsky.plot()
    piyavsky.step()
minimum, min_value, elapsed_time, iterations = piyavsky.solve()

print(f"Минимум найден: x = {minimum}, f(x) = {min_value}")
print(f"Число итераций: {iterations}")
print(f"Время выполнения: {elapsed_time:.4f} секунд")
piyavsky.plot(show_solution=True)
