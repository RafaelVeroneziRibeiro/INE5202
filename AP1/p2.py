def horner(coefs, x0):

    n = len(coefs) - 1

    y = coefs[n]
    z = coefs[n]

    for j in range(n-1, 0, -1):
        y = x0 * y + coefs[j]
        z = x0 * z + y

    y = x0 * y + coefs[0]

    return y, z


def newton_polinomio(coefs, x0, tol=1e-6, max_iter=100):
    x = x0

    for i in range(max_iter):
        fx, dfx = horner(coefs, x)

        if abs(fx) < tol:
            return x

        if dfx == 0:
            raise ValueError("Derivada zero!")

        x = x - fx / dfx

    return x



if __name__ == "__main__":
    coefs = [-4, 3, -3, 0, 2]  # a0, a1, ..., an
    x0 = -2

    raiz = newton_polinomio(coefs, x0)
    print("Raiz aproximada:", raiz)