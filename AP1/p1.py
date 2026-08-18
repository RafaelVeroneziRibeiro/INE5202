def falsa_posicao(f, a, b, tol=1e-6, max_iter=1000):
    if f(a) * f(b) >= 0:
        raise ValueError("O intervalo não contém raiz.")

    for i in range(max_iter):
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        
        if abs(f(c)) < tol:
            return c
        
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

    return c


# Função f(x) = x^2 - alpha
def raiz_quadrada(alpha):
    f = lambda x: x**2 - alpha
    
    a = 0
    b = max(1, alpha)
    
    return falsa_posicao(f, a, b)


# Teste
alpha = 5
print("Aproximação:", raiz_quadrada(alpha))