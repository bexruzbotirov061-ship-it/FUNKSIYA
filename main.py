import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def inverse_sigmoid(y_val):
    y_scaled = np.clip(y_val / 120000, 0.001, 0.999)
    return 50 + 10 * np.log(y_scaled / (1 - y_scaled))


def main():
    data = {
        'Uy_maydoni': [10, 25, 40, 50, 60, 75, 90],
        'Uy_narxi': [5000, 12000, 50000, 80000, 110000, 118000, 120000]
    }
    df = pd.DataFrame(data)

    X = df['Uy_maydoni'].values
    y = df['Uy_narxi'].values

    X_line = np.linspace(0, 100, 200)
    X_scaled = (X_line - 50) / 10
    y_line = 120000 * sigmoid(X_scaled)

    try:
        kiritilgan_narx = float(input("Uy narxini kiriting ($): "))
    except ValueError:
        print("Xatolik: Iltimos, faqat raqam kiriting!")
        return

    kalkulyatsiya_maydoni = inverse_sigmoid(kiritilgan_narx)
    print(f"\n{kiritilgan_narx}$ narxga mos keladigan uy maydoni: {kalkulyatsiya_maydoni:.2f} m²")

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color='blue', label="Jadvaldagi ma'lumotlar", s=80, zorder=5)
    plt.plot(X_line, y_line, color='red', linewidth=2.5, label="Sigmoid egri chizig'i")
    plt.scatter(
        kalkulyatsiya_maydoni, 
        kiritilgan_narx, 
        color='green', 
        marker='*', 
        s=220, 
        zorder=6, 
        label=f"Kiritilgan narx ({kiritilgan_narx}$)"
    )

    plt.xlabel("Uy maydoni (m²)")
    plt.ylabel("Uy narxi ($)")
    plt.title("Kiritilgan narx bo'yicha maydonni aniqlash")
    plt.legend()
    plt.grid(True, linestyle='-', alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()
