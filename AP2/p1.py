import numpy as np

def solve_tridiagonal(a, d, c, b):
    """
    Resolve um sistema tridiagonal pelo Algoritmo de Eliminação de Gauss.
    a: subdiagonal (a[0] não usado, índices 1..n-1)
    d: diagonal principal (índices 0..n-1)
    c: superdiagonal (índices 0..n-2, c[n-1] não usado)
    b: vetor do lado direito
    """
    n = len(d)
    # Cópias para não modificar os originais
    d = d.copy().astype(float)
    b = b.copy().astype(float)
    a = a.copy().astype(float)
    c = c.copy().astype(float)

    # Eliminação (forward)
    for k in range(n - 1):
        fator = a[k + 1] / d[k]
        d[k + 1] -= fator * c[k]
        b[k + 1] -= fator * b[k]

    # Substituição retroativa (back substitution)
    x = np.zeros(n)
    x[-1] = b[-1] / d[-1]
    for k in range(n - 2, -1, -1):
        x[k] = (b[k] - c[k] * x[k + 1]) / d[k]

    return x


# ─────────────────────────────────────────────
# (a) Teste com o sistema da equação (1)
# ─────────────────────────────────────────────
print("=" * 50)
print("Parte (a): Sistema tridiagonal de teste")
print("=" * 50)

#   3x1 + x2            = 4
#   2x1 + 4x2 +  x3     = 7
#        2x2 + 5x3 - x4 = 6
#              -x3 + 2x4 = 1

n = 4
d = np.array([3.0, 4.0, 5.0, 2.0])   # diagonal principal
c = np.array([1.0, 1.0, -1.0, 0.0])  # superdiagonal (c[n-1] não usado)
a = np.array([0.0, 2.0, 2.0, -1.0])  # subdiagonal  (a[0] não usado)
b = np.array([4.0, 7.0, 6.0, 1.0])   # lado direito

x = solve_tridiagonal(a, d, c, b)
print(f"Solução encontrada : {x}")
print(f"Solução esperada   : [1. 1. 1. 1.]")
print(f"Erro máximo        : {np.max(np.abs(x - 1.0)):.2e}\n")


# ─────────────────────────────────────────────
# (b) Distribuição de temperatura em uma barra
# ─────────────────────────────────────────────
print("=" * 50)
print("Parte (b): Temperatura na barra")
print("=" * 50)

def temperatura_barra(n):
    """
    Resolve o sistema de temperatura para n divisões da barra.
    Sistema:
        2T1  - T2           = 50
       -T_{i-1} + 2T_i - T_{i+1} = 0,  i = 2..n-2
                 -T_{n-2} + 2T_{n-1} = 0
    """
    m = n - 1  # número de incógnitas (nós internos)
    d = np.full(m, 2.0)
    c = np.full(m, -1.0)   # superdiagonal
    a = np.full(m, -1.0)   # subdiagonal
    b = np.zeros(m)
    b[0] = 50.0            # condição de contorno

    T = solve_tridiagonal(a, d, c, b)
    return T

for n in [5, 10, 20]:
    T = temperatura_barra(n)
    print(f"\nn = {n} divisões  →  {n-1} nós internos")
    print(f"Temperaturas T: {np.round(T, 4)}")