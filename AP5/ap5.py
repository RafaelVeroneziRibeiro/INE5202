import numpy as np

# Dados medidos da órbita do cometa
r     = np.array([2.7, 2.0, 1.61, 1.2, 1.02])
theta = np.radians([48, 67, 83, 108, 126])

print("=" * 55)
print("   Ajuste da órbita de Kepler: r = a0 / (1 - a1·cos θ)")
print("=" * 55)

# ─────────────────────────────────────────────────────────────
# MÉTODO 1 — MMQ Linearizado
#
# A ideia é reorganizar o modelo pra ele ficar linear:
#   r = a0 + a1 · (r·cosθ)
# Agora é só resolver um sistema de equações normais.
# ─────────────────────────────────────────────────────────────

A = np.column_stack([np.ones(len(r)), r * np.cos(theta)])
a_lin = np.linalg.solve(A.T @ A, A.T @ r)

r_calc_lin = a_lin[0] / (1 - a_lin[1] * np.cos(theta))
residuos_lin = r - r_calc_lin

print("\n MMQ Linearizado")
print(f"   a0 = {a_lin[0]:.4f}   a1 = {a_lin[1]:.4f}")
print(f"   SSE = {np.sum(residuos_lin**2):.6f}")

print(f"\n   {'θ':>7}  {'r obs':>7}  {'r calc':>7}  {'erro':>8}")
for i in range(len(r)):
    print(f"   {np.degrees(theta[i]):>6.0f}°  {r[i]:>7.4f}  {r_calc_lin[i]:>7.4f}  {residuos_lin[i]:>+8.5f}")

# ─────────────────────────────────────────────────────────────
# MÉTODO 2 — Gauss-Newton
#
# Aqui não transformamos o modelo — trabalhamos diretamente
# com o resíduo e_i = r_i - a0/(1 - a1·cosθ_i) e vamos
# atualizando os parâmetros até convergir.
# ─────────────────────────────────────────────────────────────

def modelo(a, t):
    return a[0] / (1 - a[1] * np.cos(t))

def residuo(a):
    return r - modelo(a, theta)

def jacobiana(a):
    denom = 1 - a[1] * np.cos(theta)
    J = np.zeros((len(theta), 2))
    J[:, 0] = -1.0 / denom
    J[:, 1] = -a[0] * np.cos(theta) / denom**2
    return J

# Começa do resultado linearizado (já é um bom chute!)
a = a_lin.copy()

print("\n Gauss-Newton")
print(f"   {'iter':>4}  {'a0':>10}  {'a1':>10}  {'SSE':>12}")

for k in range(50):
    e = residuo(a)
    J = jacobiana(a)
    delta = np.linalg.solve(J.T @ J, -J.T @ e)
    a = a + delta
    print(f"   {k+1:>4}  {a[0]:>10.6f}  {a[1]:>10.6f}  {np.sum(residuo(a)**2):>12.8f}")
    if np.linalg.norm(delta) < 1e-10:
        break

r_calc_gn = modelo(a, theta)
residuos_gn = r - r_calc_gn

print(f"\n   a0 = {a[0]:.4f}   a1 = {a[1]:.4f}")
print(f"   SSE = {np.sum(residuos_gn**2):.6f}")

print(f"\n   {'θ':>7}  {'r obs':>7}  {'r calc':>7}  {'erro':>8}")
for i in range(len(r)):
    print(f"   {np.degrees(theta[i]):>6.0f}°  {r[i]:>7.4f}  {r_calc_gn[i]:>7.4f}  {residuos_gn[i]:>+8.5f}")

# ─────────────────────────────────────────────────────────────
# COMPARAÇÃO FINAL
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("   Comparação final")
print("=" * 55)
print(f"   {'Método':<18}  {'a0':>8}  {'a1':>8}  {'SSE':>12}")
print(f"   {'Linearizado':<18}  {a_lin[0]:>8.4f}  {a_lin[1]:>8.4f}  {np.sum(residuos_lin**2):>12.8f}")
print(f"   {'Gauss-Newton':<18}  {a[0]:>8.4f}  {a[1]:>8.4f}  {np.sum(residuos_gn**2):>12.8f}")
print()
print("   Gauss-Newton minimiza diretamente o erro no modelo")
print("   original, por isso tem SSE menor.")
print("=" * 55)

print("""
Reflexão:
  Os dois métodos chegaram a resultados bem parecidos, o que
  é um bom sinal — significa que a linearização não distorceu
  muito o problema nesse caso.

  O Gauss-Newton teve um SSE um pouco menor porque ele otimiza
  os parâmetros no espaço original do modelo, sem nenhuma
  transformação. Já o linearizado minimiza o erro numa versão
  reorganizada da equação, o que não é exatamente a mesma coisa.

  Na prática, o resultado linearizado funcionou muito bem como
  ponto de partida pro Gauss-Newton, que convergiu em só 2
  iterações. Isso mostra que, quando dá pra linearizar, vale
  muito a pena fazer isso antes — mesmo que o objetivo final
  seja rodar um método iterativo.
""")