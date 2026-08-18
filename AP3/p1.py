import numpy as np

def sor(n, omega, tol=1e-6, max_iter=10000):
    # Inicializa o vetor x com zeros
    x = np.zeros(n)
    # Define o vetor b (1 no início e no fim, 0 no meio)
    b = np.zeros(n)
    b[0], b[-1] = 1, 1
    
    for iteration in range(max_iter):
        x_old = np.copy(x)
        
        for i in range(n):
            # Soma dos termos conhecidos (vizinhos na matriz tridiagonal)
            sigma = 0
            if i > 0:
                sigma += -1 * x[i-1] # Termo da iteração atual (já calculado)
            if i < n - 1:
                sigma += -1 * x[i+1] # Termo da iteração anterior
            
            # Aplicação da fórmula SOR
            x[i] = (1 - omega) * x[i] + (omega / 2.0) * (b[i] - sigma)
            
        # Critério de parada: norma do erro
        if np.linalg.norm(x - x_old, np.inf) < tol:
            return iteration + 1
            
    return max_iter

# --- Script Principal ---
n = int(input("Digite o tamanho do sistema (n): "))

# 1. Testar Gauss-Seidel (omega = 1)
iter_gs = sor(n, 1.0)
print(f"Iterações com Gauss-Seidel (omega=1): {iter_gs}")

# 2. Busca experimental pelo omega ótimo
best_omega = 1.0
min_iters = iter_gs

print("\nBuscando omega ótimo...")
for w in np.arange(1.0, 2.0, 0.02):
    iters = sor(n, w)
    if iters < min_iters:
        min_iters = iters
        best_omega = w

print(f"Omega ótimo aproximado: {best_omega:.2f}")
print(f"Número de iterações com omega ótimo: {min_iters}")