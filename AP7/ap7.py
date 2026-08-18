import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# QUESTÃO 1: PVI dy/dt = -t * y
# =============================================================================

def f1(t, y):
    """Função do PVI da Questão 1."""
    return -t * y

def solucao_exata(t):
    """Solução exata analítica para comparação."""
    return np.exp(-(t**2) / 2)

def euler_rk1(f, t0, tf, y0, dt):
    """Método de Euler Explícito (RK1)."""
    N = int((tf - t0) / dt)
    t = np.linspace(t0, tf, N + 1)
    y = np.zeros(N + 1)
    y[0] = y0
    for k in range(N):
        y[k+1] = y[k] + dt * f(t[k], y[k])
    return t, y

def euler_aprimorado_rk2(f, t0, tf, y0, dt):
    """Método de Euler Aprimorado (RK2)."""
    N = int((tf - t0) / dt)
    t = np.linspace(t0, tf, N + 1)
    y = np.zeros(N + 1)
    y[0] = y0
    for k in range(N):
        K1 = f(t[k], y[k])
        K2 = f(t[k] + dt, y[k] + dt * K1)
        y[k+1] = y[k] + (dt / 2.0) * (K1 + K2)
    return t, y

def rk4_escalar(f, t0, tf, y0, dt):
    """Método de Runge-Kutta de 4ª Ordem (RK4) para uma equação."""
    N = int((tf - t0) / dt)
    t = np.linspace(t0, tf, N + 1)
    y = np.zeros(N + 1)
    y[0] = y0
    for k in range(N):
        K1 = f(t[k], y[k])
        K2 = f(t[k] + dt/2.0, y[k] + (dt/2.0) * K1)
        K3 = f(t[k] + dt/2.0, y[k] + (dt/2.0) * K2)
        K4 = f(t[k] + dt, y[k] + dt * K3)
        y[k+1] = y[k] + (dt / 6.0) * (K1 + 2*K2 + 2*K3 + K4)
    return t, y

# --- Execução e análise de erro para a Questão 1 ---
passos_h = [0.1, 0.05, 0.025, 0.01, 0.005]
t0, tf, y0 = 0.0, 1.0, 1.0
y_exata_tf = solucao_exata(tf)

print("=== QUESTÃO 1: Tabela de Erros em t = 1 ===")
print(f"{'dt (h)':<8} | {'Erro RK1':<12} | {'Erro RK2':<12} | {'Erro RK4':<12}")
print("-" * 55)

for dt in passos_h:
    _, y_rk1 = euler_rk1(f1, t0, tf, y0, dt)
    _, y_rk2 = euler_aprimorado_rk2(f1, t0, tf, y0, dt)
    _, y_rk4 = rk4_escalar(f1, t0, tf, y0, dt)
    
    err_rk1 = abs(y_exata_tf - y_rk1[-1])
    err_rk2 = abs(y_exata_tf - y_rk2[-1])
    err_rk4 = abs(y_exata_tf - y_rk4[-1])
    
    print(f"{dt:<8.4f} | {err_rk1:<12.4e} | {err_rk2:<12.4e} | {err_rk4:<12.4e}")
print("\n" + "="*55 + "\n")


# =============================================================================
# QUESTÃO 2: Sistema Lotka-Volterra
# =============================================================================

# Parâmetros do sistema
alpha = 3.0
beta = 1.0

def F_sistema(t, vetor):
    """Função vetorial que define o sistema Lotka-Volterra."""
    x, y = vetor[0], vetor[1]
    dxdt = alpha * x * (1.0 - y)
    dydt = beta * y * (x - 1.0)
    return np.array([dxdt, dydt])

def euler_sistema(F, t0, tf, vetor0, dt):
    """Euler Explícito para sistemas de EDOs."""
    N = int((tf - t0) / dt)
    t = np.linspace(t0, tf, N + 1)
    vetor = np.zeros((N + 1, 2))
    vetor[0] = vetor0
    for k in range(N):
        vetor[k+1] = vetor[k] + dt * F(t[k], vetor[k])
    return t, vetor

def rk4_sistema(F, t0, tf, vetor0, dt):
    """RK4 para sistemas de EDOs."""
    N = int((tf - t0) / dt)
    t = np.linspace(t0, tf, N + 1)
    vetor = np.zeros((N + 1, 2))
    vetor[0] = vetor0
    for k in range(N):
        v_k = vetor[k]
        K1 = F(t[k], v_k)
        K2 = F(t[k] + dt/2.0, v_k + (dt/2.0) * K1)
        K3 = F(t[k] + dt/2.0, v_k + (dt/2.0) * K2)
        K4 = F(t[k] + dt, v_k + dt * K3)
        vetor[k+1] = v_k + (dt / 6.0) * (K1 + 2*K2 + 2*K3 + K4)
    return t, vetor

# Parâmetros de simulação do sistema
t0_sys, tf_sys = 0.0, 15.0  # Simulação até t=15 para melhor visualização das órbitas
dt_sys = 0.005              # Passo refinado para garantir a estabilidade do Euler
condicoes_iniciais = np.array([3.0, 1.0])

# Resolvendo o sistema por ambos os métodos
t_sys, res_rk1 = euler_sistema(F_sistema, t0_sys, tf_sys, condicoes_iniciais, dt_sys)
_, res_rk4 = rk4_sistema(F_sistema, t0_sys, tf_sys, condicoes_iniciais, dt_sys)

# Separando as variáveis x(t) e y(t)
x_rk1, y_rk1 = res_rk1[:, 0], res_rk1[:, 1]
x_rk4, y_rk4 = res_rk4[:, 0], res_rk4[:, 1]

# --- Construção dos Gráficos ---
plt.figure(figsize=(14, 6))

# Gráfico 1: Evolução Temporal x(t) e y(t)
plt.subplot(1, 2, 1)
plt.plot(t_sys, x_rk1, 'r--', label='x(t) - Euler (RK1)', alpha=0.7)
plt.plot(t_sys, y_rk1, 'b--', label='y(t) - Euler (RK1)', alpha=0.7)
plt.plot(t_sys, x_rk4, 'r-', linewidth=2, label='x(t) - RK4')
plt.plot(t_sys, y_rk4, 'b-', linewidth=2, label='y(t) - RK4')
plt.xlabel('Tempo (t)')
plt.ylabel('Populações')
plt.title('Evolução Temporal das Populações')
plt.legend()
plt.grid(True)

# Gráfico 2: Retrato de Fase x(t) vs y(t)
plt.subplot(1, 2, 2)
plt.plot(x_rk1, y_rk1, 'g--', label='Órbita - Euler (RK1)', alpha=0.7)
plt.plot(x_rk4, y_rk4, 'm-', linewidth=2, label='Órbita - RK4')
plt.scatter(condicoes_iniciais[0], condicoes_iniciais[1], color='black', zorder=5, label='Ponto Inicial (3,1)')
plt.xlabel('População x(t)')
plt.ylabel('População y(t)')
plt.title('Retrato de Fase: x(t) vs y(t)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()