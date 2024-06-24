#Nossa IYAMU, Louis SIMON

import numpy as np

#EXERCICE 1

def simuV(v, s, r):
    dt = 0.1
    T = 10
    N = int(T / dt)
    V = np.zeros(N + 1)
    V[0] = 1
    B = np.random.normal(0, 1, N)
    W = r * B + np.sqrt(1 - r**2) * np.random.normal(0, 1, N)
    
    for t in range(1, N + 1):
        V_t = V[t - 1] + (v - V[t - 1]) * dt + s * np.sqrt(max(V[t - 1], 0)) * np.sqrt(dt) * W[t - 1]
        V[t] = max(V_t, 0)
        
    return V
    
def simuX(v, s, r):
    dt = 0.1
    T = 10
    N = int(T / dt)
    X = np.zeros(N + 1)
    X[0] = 100
    V = simuV(v, s, r)
    B = np.random.normal(0, 1, N)
    W = r * B + np.sqrt(1 - r**2) * np.random.normal(0, 1, N)
    
    for t in range(1, N + 1):
        X[t] = X[t - 1] + X[t - 1] * (0.01 * dt + 0.01 * np.sqrt(V[t - 1]) * np.sqrt(dt) * B[t - 1])
        
    return X

def optionAsia(v, s, r):
    samples = 10000
    asian_options = np.zeros(samples)
    
    for i in range(samples):
        X = simuX(v, s, r)
        asian_options[i] = max(np.mean(X) - 100, 0)
        
    return np.mean(asian_options)

def supV(v, s, r):
    V = simuV(v, s, r)
    return np.max(V)


#EXERCICE 2

def simuEuro(x, m):
    T = 120
    B = np.random.normal(0, 1, T)
    S = np.zeros(T + 1)
    
    for t in range(1, T + 1):
        S[t] = x * np.exp(B[t] / 10 + (m - 0.5) * t / 10)
        
    return max(S[-1] - x, 0)

def simuDouble(x, m):
    T = 120
    B = np.random.normal(0, 1, T)
    S = np.zeros(T + 1)
    
    for t in range(1, T + 1):
        S[t] = x * np.exp(B[t] / 10 + (m - 0.5) * t / 10)
        
    for t in range(6, T + 1, 6):
        if S[t] >= 2 * x:
            return S[t] - x
        
    return max(S[-1] - x, 0)

def simuSomme(x, m):
    T = 120
    B = np.random.normal(0, 1, T)
    S = np.zeros(T + 1)
    
    for t in range(1, T + 1):
        S[t] = x * np.exp(B[t] / 10 + (m - 0.5) * t / 10)
        
    for t in range(12, T + 1, 6):
        if S[t] >= (S[t - 6] + S[t - 12]):
            return S[t] - x
        
    return max(S[-1] - x, 0)

# EXERCICE 3
def euler_maruyama(h, T, mu, sigma, X0):
    N = int(T / h)
    X = np.zeros(N + 1)
    X[0] = X0
    B = np.random.normal(0, np.sqrt(h), N)
    
    for t in range(1, N + 1):
        X[t] = X[t - 1] + mu(X[t - 1]) * h + sigma(X[t - 1]) * B[t - 1]
        
    return X

def milstein(h, T, mu, sigma, X0):
    N = int(T / h)
    X = np.zeros(N + 1)
    X[0] = X0
    B = np.random.normal(0, np.sqrt(h), N)
    
    for t in range(1, N + 1):
        dB = B[t - 1]
        X[t] = X[t - 1] + mu(X[t - 1]) * h + sigma(X[t - 1]) * dB + 0.5 * sigma(X[t - 1]) * sigma(X[t - 1]) * (dB * dB - h)
        
    return X

def mu(X):
    return 1 - X

def sigma(X):
    return np.sqrt(X)

def compare_methods():
    T = 1
    X0 = 1
    h1 = 0.01
    h2 = 0.0001
    h3 = 0.01
    
    euler_maruyama_1 = np.mean([euler_maruyama(h1, T, mu, sigma, X0)[-1]**2 for _ in range(100)])
    euler_maruyama_2 = np.mean([euler_maruyama(h2, T, mu, sigma, X0)[-1]**2 for _ in range(100)])
    milstein_1 = np.mean([milstein(h3, T, mu, sigma, X0)[-1]**2 for _ in range(100)])
    
    return euler_maruyama_1, euler_maruyama_2, milstein_1