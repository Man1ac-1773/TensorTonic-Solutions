def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    for i in range(steps):
        x0 -= deriv(a,b,c,x0) * lr
    return x0

def deriv(a,b,c,x):
    return (2*a*x+b)
        