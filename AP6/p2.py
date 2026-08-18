import math

def regra_trapezios(f, a, b, n):
    """
    Aproxima a integral de f(x) de 'a' até 'b' usando n subintervalos.
    """
    h = (b - a) / n
    soma = 0.5 * (f(a) + f(b))
    
    for i in range(1, n):
        x_i = a + i * h
        soma += f(x_i)
        
    return soma * h

# Parâmetros do problema
a = 1.0
b = 3.0
n = 20  # Você pode alterar a quantidade de subintervalos aqui
f = lambda x: 1.0 / x

# Calculando a aproximação
aproximacao = regra_trapezios(f, a, b, n)

# Calculando o erro cometido
valor_exato = math.log(3)
erro = abs(valor_exato - aproximacao)

print("--- Problema 2: Regra dos Trapézios ---")
print(f"Intervalo: [{a}, {b}], Subintervalos (n): {n}")
print(f"Aproximação da integral: {aproximacao:.6f}")
print(f"Valor exato de ln(3): {valor_exato:.6f}")
print(f"Erro cometido: {erro:.6e}")