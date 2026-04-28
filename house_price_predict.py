def load_training_data():
    x_trains = []
    y_trains = []
    with open("data.txt", "r") as f:
        for line in f:
            parts = line.strip().split(",")
            col1, col2, col3 = float(parts[0]), float(parts[1]), float(parts[2])
            x_trains.append([col1 / 1000.0, col2])
            y_trains.append(col3 / 1000.0)
    return x_trains, y_trains

def main():
    x, y = load_training_data()
    m = len(y)

    w = [0.0, 0.0]
    b = 0.0
    lr = 1e-2

    iter_count = int(input("Enter number of iterations: "))

    for i in range(iter_count):
        # forward
        e = [0.0] * m
        for j in range(m):
            e[j] = (w[0] * x[j][0] + w[1] * x[j][1] + b) - y[j]

        loss = sum(e[k] ** 2 for k in range(m)) / (2.0 * m)

        print(f"Iteration {i}, Loss: {loss:.6f}, w1: {w[0]:.6f}, w2: {w[1]:.6f}, b: {b:.6f}")

        # backward
        dl_dw = [0.0, 0.0]
        for j in range(m):
            dl_dw[0] += e[j] * x[j][0] / m
            dl_dw[1] += e[j] * x[j][1] / m
        dl_db = sum(e) / m

        w[0] -= lr * dl_dw[0]
        w[1] -= lr * dl_dw[1]
        b -= lr * dl_db

    print(f"\nFinal w1: {w[0]:.6f}, w2: {w[1]:.6f}, b: {b:.6f}")

if __name__ == "__main__":
    main()