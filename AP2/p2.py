import numpy as np
from scipy.linalg import solve, lu, det
from numpy.linalg import cond

# ─────────────────────────────────────────────
# Funções auxiliares
# ─────────────────────────────────────────────

def hilbert(n):
    """Constrói a matriz de Hilbert H_n onde h_ij = 1/(i+j-1)."""
    H = np.array([[1.0 / (i + j - 1) for j in range(1, n + 1)]
                  for i in range(1, n + 1)])
    return H

def vetor_b(n):
    """Constrói b_i = sum_{j=1}^{n} 1/(i+j-1)  (soma da linha i de H_n)."""
    H = hilbert(n)
    return H.sum(axis=1)


# ─────────────────────────────────────────────
# (a) Resolução de H_n x = b  com solve()
# ─────────────────────────────────────────────
print("=" * 60)
print("Parte (a): Resolução dos sistemas H_n x = b")
print("(solução exata = vetor de 1's)")
print("=" * 60)

for n in [3, 5, 10, 20]:
    H = hilbert(n)
    b = vetor_b(n)
    x = solve(H, b)
    erro = np.max(np.abs(x - 1.0))
    print(f"\nn = {n}")
    print(f"  Solução x : {np.round(x, 6)}")
    print(f"  Erro máx  : {erro:.2e}")


# ─────────────────────────────────────────────
# (b) Fatoração LU de H_n com lu()
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Parte (b): Fatoração LU de H_n")
print("=" * 60)

for n in [3, 5, 10]:
    H = hilbert(n)
    P, L, U = lu(H)
    print(f"\nn = {n}")
    print(f"  Matriz P (permutação):\n{np.round(P, 4)}")
    print(f"  Matriz L (triangular inferior):\n{np.round(L, 4)}")
    print(f"  Matriz U (triangular superior):\n{np.round(U, 6)}")
    # Verificação: P @ L @ U deve reconstruir H
    erro_lu = np.max(np.abs(P @ L @ U - H))
    print(f"  Erro de reconstrução P·L·U ≈ H : {erro_lu:.2e}")


# ─────────────────────────────────────────────
# (c) Número de condição e determinante de H_n
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Parte (c): Condicionamento das matrizes de Hilbert")
print("=" * 60)
print(f"{'n':>4}  {'cond(H_n)':>18}  {'det(H_n)':>18}  {'Mal cond.?':>10}")
print("-" * 60)

for n in [3, 5, 10, 20]:
    H = hilbert(n)
    c_num = cond(H)
    d_num = det(H)
    mal = "SIM" if c_num > 1e6 else "não"
    print(f"{n:>4}  {c_num:>18.4e}  {d_num:>18.4e}  {mal:>10}")

print("""
Observação:
  - Número de condição alto (>> 1) indica sistema mal condicionado.
  - Determinante próximo de zero reforça o diagnóstico.
  - As matrizes de Hilbert tornam-se progressivamente mais mal
    condicionadas à medida que n cresce.
""")