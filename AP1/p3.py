import math

def newton(f, df, x0, tol=1e-6, max_iter=100):
    x = x0

    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)

        if abs(fx) < tol:
            return x, i + 1

        if dfx == 0:
            raise ValueError("Derivada zero!")

        x = x - fx / dfx

    return x, max_iter


def newton_modificado(f, df, ddf, x0, tol=1e-6, max_iter=100):
    x = x0

    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        ddfx = ddf(x)

        if abs(fx) < tol:
            return x, i + 1

        denominador = (dfx**2 - fx * ddfx)

        if denominador == 0:
            raise ValueError("Divisão por zero!")

        x = x - (fx * dfx) / denominador

    return x, max_iter



if __name__ == "__main__":

    print("===== EXEMPLO 1 =====")

    f1 = lambda x: math.exp(x) - x - 1
    df1 = lambda x: math.exp(x) - 1
    ddf1 = lambda x: math.exp(x)

    x0_1 = (0.5 + 1.5) / 2

    r1, it1 = newton(f1, df1, x0_1)
    r2, it2 = newton_modificado(f1, df1, ddf1, x0_1)

    print("Newton:", it1, "iterações")
    print("Modificado:", it2, "iterações")


    print("\n===== EXEMPLO 2 =====")

    f2 = lambda x: math.exp(3*x) - 27*x**6 + 27*x**4*math.exp(x) - 9*x**2*math.exp(2*x)
    df2 = lambda x: (
        3*math.exp(3*x)
        - 162*x**5
        + 108*x**3*math.exp(x)
        + 27*x**4*math.exp(x)
        - 18*x*math.exp(2*x)
        - 18*x**2*math.exp(2*x)
    )
    ddf2 = lambda x: (
        9*math.exp(3*x)
        - 810*x**4
        + 324*x**2*math.exp(x)
        + 216*x**3*math.exp(x)
        + 27*x**4*math.exp(x)
        - 18*math.exp(2*x)
        - 72*x*math.exp(2*x)
        - 36*x**2*math.exp(2*x)
    )

    x0_2 = (3 + 5) / 2

    r1, it1 = newton(f2, df2, x0_2)
    r2, it2 = newton_modificado(f2, df2, ddf2, x0_2)

    print("Newton:", it1, "iterações")
    print("Modificado:", it2, "iterações")