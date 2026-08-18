"""
Sistema:
    f1(x1, x2) = x1² - 10*x1 + x2² + 8 = 0
    f2(x1, x2) = x1*x2² + x1 - 10*x2 + 8 = 0

Solução exata (dentro de D): x1 = x2 = 1  (verificável por substituição)
"""

import numpy as np

# ─────────────────────────────────────────────
# Definições do sistema
# ─────────────────────────────────────────────

def F(x):
    """Vetor de funções do sistema."""
    x1, x2 = x
    return np.array([
        x1**2 - 10*x1 + x2**2 + 8,
        x1*x2**2 + x1 - 10*x2 + 8
    ])

def J(x):
    """Matriz Jacobiana de F."""
    x1, x2 = x
    return np.array([
        [2*x1 - 10,      2*x2        ],
        [x2**2 + 1,   2*x1*x2 - 10  ]
    ])

def G(x):
    """Função de iteração de ponto fixo."""
    x1, x2 = x
    return np.array([
        (x1**2 + x2**2 + 8) / 10,
        (x1*x2**2 + x1 + 8) / 10
    ])

def G_gauss_seidel(x):
    """
    Variante Gauss-Seidel: usa o x1 atualizado imediatamente no cálculo de x2.
    """
    x1, x2 = x
    x1_novo = (x1**2 + x2**2 + 8) / 10          # atualiza x1
    x2_novo = (x1_novo * x2**2 + x1_novo + 8) / 10  # usa x1_novo
    return np.array([x1_novo, x2_novo])

# ─────────────────────────────────────────────
# Critério de parada
# ─────────────────────────────────────────────

def norma_infinito(v):
    return np.max(np.abs(v))

# ─────────────────────────────────────────────
# 1a) Aproximações Sucessivas (Ponto Fixo)
# ─────────────────────────────────────────────

def aproximacoes_sucessivas(x0, tol=1e-6, max_iter=100):
    """
    Método das Aproximações Sucessivas (iteração de ponto fixo).
    x^(k+1) = G(x^(k))
    """
    print("=" * 55)
    print("1a) APROXIMAÇÕES SUCESSIVAS (Ponto Fixo)")
    print("=" * 55)
    print(f"{'Iter':>5}  {'x1':>14}  {'x2':>14}  {'||erro||':>14}")
    print("-" * 55)

    x = x0.copy()
    historico = []

    for k in range(max_iter):
        x_novo = G(x)
        erro = norma_infinito(x_novo - x)
        historico.append((k+1, x_novo.copy(), erro))
        print(f"{k+1:>5}  {x_novo[0]:>14.8f}  {x_novo[1]:>14.8f}  {erro:>14.2e}")

        if erro < tol:
            print(f"\n  Convergiu em {k+1} iterações.")
            print(f"  Solução: x1 = {x_novo[0]:.8f}, x2 = {x_novo[1]:.8f}")
            print(f"  ||F(x)|| = {norma_infinito(F(x_novo)):.2e}\n")
            return x_novo, historico

        x = x_novo

    print("  AVISO: máximo de iterações atingido sem convergência.\n")
    return x, historico

# ─────────────────────────────────────────────
# 1b) Gauss-Seidel (Ponto Fixo acelerado)
# ─────────────────────────────────────────────

def gauss_seidel_ponto_fixo(x0, tol=1e-6, max_iter=100):
    """
    Técnica de Gauss-Seidel aplicada à iteração de ponto fixo:
    usa o componente atualizado assim que disponível.
    """
    print("=" * 55)
    print("1b) GAUSS-SEIDEL (Ponto Fixo)")
    print("=" * 55)
    print(f"{'Iter':>5}  {'x1':>14}  {'x2':>14}  {'||erro||':>14}")
    print("-" * 55)

    x = x0.copy()
    historico = []

    for k in range(max_iter):
        x_novo = G_gauss_seidel(x)
        erro = norma_infinito(x_novo - x)
        historico.append((k+1, x_novo.copy(), erro))
        print(f"{k+1:>5}  {x_novo[0]:>14.8f}  {x_novo[1]:>14.8f}  {erro:>14.2e}")

        if erro < tol:
            print(f"\n  Convergiu em {k+1} iterações.")
            print(f"  Solução: x1 = {x_novo[0]:.8f}, x2 = {x_novo[1]:.8f}")
            print(f"  ||F(x)|| = {norma_infinito(F(x_novo)):.2e}\n")
            return x_novo, historico

        x = x_novo

    print("  AVISO: máximo de iterações atingido sem convergência.\n")
    return x, historico

# ─────────────────────────────────────────────
# 2) Método de Newton
# ─────────────────────────────────────────────

def newton(x0, tol=1e-6, max_iter=50):
    """
    Método de Newton para sistemas não lineares.
    Resolve J(x^(k)) * delta = -F(x^(k))  →  x^(k+1) = x^(k) + delta
    """
    print("=" * 55)
    print("2) MÉTODO DE NEWTON")
    print("=" * 55)
    print(f"{'Iter':>5}  {'x1':>14}  {'x2':>14}  {'||F(x)||':>14}")
    print("-" * 55)

    x = x0.copy()
    historico = []

    for k in range(max_iter):
        Fx = F(x)
        Jx = J(x)
        norma_F = norma_infinito(Fx)
        historico.append((k, x.copy(), norma_F))
        print(f"{k:>5}  {x[0]:>14.8f}  {x[1]:>14.8f}  {norma_F:>14.2e}")

        if norma_F < tol:
            print(f"\n  Convergiu em {k} iterações.")
            print(f"  Solução: x1 = {x[0]:.8f}, x2 = {x[1]:.8f}\n")
            return x, historico

        # Resolve J * delta = -F
        delta = np.linalg.solve(Jx, -Fx)
        x = x + delta

    print("  AVISO: máximo de iterações atingido sem convergência.\n")
    return x, historico

# ─────────────────────────────────────────────
# 3) Método de Newton Modificado
# ─────────────────────────────────────────────

def newton_modificado(x0, tol=1e-6, max_iter=100):
    """
    Método de Newton Modificado:
    A Jacobiana é calculada APENAS na iteração inicial (x0) e reutilizada.
    Custo por iteração menor, mas convergência mais lenta (linear em vez de quadrática).
    """
    print("=" * 55)
    print("3) MÉTODO DE NEWTON MODIFICADO")
    print("=" * 55)
    print(f"{'Iter':>5}  {'x1':>14}  {'x2':>14}  {'||F(x)||':>14}")
    print("-" * 55)

    x = x0.copy()
    J0 = J(x0)       # Jacobiana fixa em x0
    historico = []

    for k in range(max_iter):
        Fx = F(x)
        norma_F = norma_infinito(Fx)
        historico.append((k, x.copy(), norma_F))
        print(f"{k:>5}  {x[0]:>14.8f}  {x[1]:>14.8f}  {norma_F:>14.2e}")

        if norma_F < tol:
            print(f"\n  Convergiu em {k} iterações.")
            print(f"  Solução: x1 = {x[0]:.8f}, x2 = {x[1]:.8f}\n")
            return x, historico

        # Mesma J0 em todas as iterações
        delta = np.linalg.solve(J0, -Fx)
        x = x + delta

    print("  AVISO: máximo de iterações atingido sem convergência.\n")
    return x, historico

# ─────────────────────────────────────────────
# Execução principal
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Aproximação inicial dentro de D = {(x1,x2) | 0 ≤ x1, x2 ≤ 1.5}
    x0 = np.array([0.0, 0.0])

    print(f"\nAproximação inicial: x0 = {x0}\n")

    sol_as,  hist_as  = aproximacoes_sucessivas(x0)
    sol_gs,  hist_gs  = gauss_seidel_ponto_fixo(x0)
    sol_n,   hist_n   = newton(x0)
    sol_nm,  hist_nm  = newton_modificado(x0)

    # ─── Tabela resumo ───────────────────────────────
    print("=" * 55)
    print("RESUMO COMPARATIVO")
    print("=" * 55)
    print(f"{'Método':<30}  {'Iterações':>9}  {'||F(sol)||':>12}")
    print("-" * 55)
    print(f"{'Aprox. Sucessivas':<30}  {len(hist_as):>9}  {norma_infinito(F(sol_as)):>12.2e}")
    print(f"{'Gauss-Seidel (P. Fixo)':<30}  {len(hist_gs):>9}  {norma_infinito(F(sol_gs)):>12.2e}")
    print(f"{'Newton':<30}  {len(hist_n):>9}  {norma_infinito(F(sol_n)):>12.2e}")
    print(f"{'Newton Modificado':<30}  {len(hist_nm):>9}  {norma_infinito(F(sol_nm)):>12.2e}")
    print("=" * 55)

    # ─── Questão 4: Reflexão impressa ────────────────
    print("""
4) REFLEXÃO:
   • Gauss-Seidel vs Aprox. Sucessivas (Ponto Fixo):
     Gauss-Seidel usa o valor atualizado de x1 imediatamente ao
     calcular x2, o que geralmente acelera a convergência.

   • Newton converge quadraticamente (muito rápido), mas exige
     resolver um sistema linear a cada iteração — custo O(n³).

   • Newton Modificado reutiliza a Jacobiana inicial, reduzindo
     o custo por iteração, mas perde a convergência quadrática
     (passa a ser linear). Vantajoso quando n é grande e F varia
     suavemente, pois economiza fatorações de matriz.

   • Ponto Fixo é o mais simples, porém depende da contração
     de G para convergir — não há garantia geral de convergência
     como no Newton.
""")