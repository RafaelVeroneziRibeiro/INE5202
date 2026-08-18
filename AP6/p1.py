import math

def diferenças_divididas(x, y):
    """Calcula a tabela de diferenças divididas de Newton."""
    n = len(y)
    coef = y.copy()
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (x[i] - x[i - j])
    return coef

def avalia_polinomio_newton(coef, x_data, x_alvo):
    """Avalia o polinômio na forma de Newton em um ponto x_alvo."""
    n = len(coef)
    resultado = coef[n - 1]
    for k in range(1, n):
        resultado = coef[n - 1 - k] + (x_alvo - x_data[n - 1 - k]) * resultado
    return resultado

# Dados da tabela (selecionando 5 pontos mais próximos de 3.15 para grau 4)
x_data = [2.6, 2.8, 3.0, 3.2, 3.4]
y_data = [13.46, 16.44, 20.08, 24.53, 29.96]
x_alvo = 3.15

# Obtendo os coeficientes
coeficientes = diferenças_divididas(x_data, y_data)

# Estimando o valor
estimativa = avalia_polinomio_newton(coeficientes, x_data, x_alvo)

# Calculando o erro exato
valor_exato = math.exp(x_alvo)
erro = abs(valor_exato - estimativa)

print("--- Problema 1: Interpolação de Newton ---")
print(f"Coeficientes do polinômio: {coeficientes}")
print(f"Estimativa para e^(3.15): {estimativa:.6f}")
print(f"Valor real de e^(3.15): {valor_exato:.6f}")
print(f"Erro cometido: {erro:.6e}")