import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


#1D Chaotic Maps
def logistic_map(x):
    a = 3.99
    return a * x * (1 - x)

def tent_map(x):
    a = 1.99
    return (a * x) % 1 if x > 0.5 else (a * (1 - x)) % 1

def sine_map(x):
    a = 0.97
    return a * math.sin(math.pi * x)

def cubic_map(x):
    a = 2.59
    return a * x * (1 - x**2)

def cosine_polynomial_map(x):
    a = 2.5
    return math.cos(a * (x**3 + x))

#2D Chaotic Maps
def henon_map(x, y):
    a = 1.4
    b = 0.3
    x2 = 1 - a * x**2 + y
    y2 = b * x
    return x2, y2

def lozi_map(x, y):
    a = 1.7
    b = 0.5
    x2 = 1 + y - a * abs(x)
    x2 = np.clip(x2, -1e10, 1e10)
    y2 = b * x
    y2 = np.clip(y2, -1e10, 1e10)
    return x2, y2

def gingerbread_man_map(x, y):
    x2 = 1 - y + abs(x)
    y2 = x
    return x2, y2

#6D Chaotic Map
def hyperchaotic_ode(t, x, a=10, b=8/3, c=28, d=-1, e=8, r=3):
    dx1 = a * (x[1] - x[0]) + x[3] - x[4] - x[5]
    dx2 = c * x[0] - x[1] - x[0] * x[2]
    dx3 = -b * x[2] + x[0] * x[1]
    dx4 = d * x[3] - x[1] * x[2]
    dx5 = e * x[5] + x[2] * x[1]
    dx6 = r * x[0]
    return [dx1, dx2, dx3, dx4, dx5, dx6]



#Encrypt Method
def Encrypt(I, rounds=2):
    dim = 6  # 1 for 1D maps, 2 for 2D maps, 6 for hyperchaotic map
    
    I = I.astype(np.uint8)
    M, N = I.shape

    if M % 2 == 1:
        I = np.vstack((I, np.zeros((1, N), dtype=np.uint8)))
        M += 1
    if N % 2 == 1:
        I = np.hstack((I, np.zeros((M, 1), dtype=np.uint8)))
        N += 1

    MN = M * N
    SS = []

    for round_iter in range(rounds):
        P = I.astype(np.float64)
        
        seq = np.zeros(MN)
        
        if dim == 1:
            P = I.flatten()
            for i in range(MN):
                x = sine_map(P[i])
                seq[i] = x
        elif dim == 2:
            for i in range(M):
                for j in range(N):  
                    x, y = gingerbread_man_map(P[i,j], P[i,j])
                    seq[i] = x+y
        elif dim == 6:
            P = I.flatten().astype(np.float64)
            x = [(np.sum(P) + MN) / (MN + (2**23))]
            for _ in range(5):
                x.append(np.mod(x[-1] * 1e6, 1.0))

            # Solve ODE
            N0 = 0.9865 * MN / 3
            MN3 = int(np.ceil(MN / 3))
            t_span = (N0, MN3)
            t_eval = np.linspace(N0, MN3, MN3)

            sol = solve_ivp(hyperchaotic_ode, t_span, x, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-9)

            # Use x1, x3, x5 from solution (columns 0, 2, 4)
            seq = sol.y[[0, 2, 4], :].T.flatten()[:MN]

        P = P.flatten()
        
        S = np.argsort(seq)
        SS.append(S)

        R = P[S]

        R_ = R.reshape(M, N)
        A = np.array([[89, 55], [55, 34]])
        C = np.zeros((M, N), dtype=np.float64)

        for i in range(0, M, 2):
            for j in range(0, N, 2):
                Cx = np.array([[R_[i, j], R_[i, j+1]],
                               [R_[i+1, j], R_[i+1, j+1]]])
                fz = np.dot(Cx, A)
                C[i, j] = fz[0, 0]
                C[i, j+1] = fz[0, 1]
                C[i+1, j] = fz[1, 0]
                C[i+1, j+1] = fz[1, 1]

        I = np.mod(C, 256).astype(np.uint8)

    return I, SS



#Decrypt Method
def Decrypt(I_enc, SS):
    
    I_enc = I_enc.astype(np.uint8)
    M, N = I_enc.shape

    C = I_enc.astype(np.float64)
    A_ = np.array([[34, -55], [-55, 89]])

    rounds = len(SS)

    for round_iter in range(rounds-1, -1, -1):
        D = np.zeros_like(C)

        for i in range (0, M, 2):
            for j in range(0, N, 2):
                Cx = np.array([
                    [C[i, j], C[i, j+1]],
                     [C[i+1, j], C[i+1, j+1]]
                ])
                fz = np.dot(Cx, A_)
                D[i, j]     = fz[0, 0]
                D[i, j+1]   = fz[0, 1]
                D[i+1, j]   = fz[1, 0]
                D[i+1, j+1] = fz[1, 1]
        
        S = SS[round_iter]
        S2 = np.argsort(S)
        W = D.flatten()
        ER = W[S2]
        C = ER.reshape((M, N))
        C = np.mod(C, 256)

    return C.astype(np.uint8)


#Usage Example
I = cv2.imread("files/boat.tiff",0)
M, N = I.shape

if M % 2 == 1:
    M += 1
if N % 2 == 1:
    N += 1

I = cv2.resize(I, (N, M))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
plt.title('Original Image')


rounds = 2
I_enc = np.zeros_like(I)

I_enc, SX = Encrypt(I, rounds)

plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(I_enc, cv2.COLOR_BGR2RGB))
plt.title('Encrypted Image')


I_dec = np.zeros_like(I_enc)
I_dec = Decrypt(I_enc, SX)

plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(I_dec, cv2.COLOR_BGR2RGB))
plt.title('Decrypted Image')



y1 = I_enc.flatten()
y2 = I_dec.flatten()
MSE = np.sum((y1 - y2) ** 2) / len(y1)

impsnr = cv2.PSNR(I_dec, I_enc)

print(f'MSE: {MSE}')
print(f'PSNR: {impsnr}')


plt.subplot(2, 3, 4)
plt.hist(I.ravel(), bins=256, color='blue')
plt.title("Histogram - Original Image")

plt.subplot(2, 3, 5)
plt.hist(I_enc.ravel(), bins=256, color='red')
plt.title("Histogram - Encrypted Image")

plt.subplot(2, 3, 6)
plt.hist(I_dec.ravel(), bins=256, color='green')
plt.title("Histogram - Decrypted Image")

plt.show()