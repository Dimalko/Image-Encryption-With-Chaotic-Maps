import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import warnings


# --- 1D Chaos Maps ---
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

# --- 2D Chaos Maps ---
def henon_map(x, y):
    a = 1.4
    b = 0.3
    x2 = 1 - a * x**2 + y
    y2 = b * x
    return max(-1, min(1, x2)), y2

def lozi_map(x, y):
    a = 1.7
    b = 0.5
    x2 = 1 + y - a * abs(x)
    y2 = b * x
    return x2, y2

def gingerbread_man_map(x, y):
    x2 = 1 - y + abs(x)
    y2 = x
    return x2, y2


#Encrypt method
def Encrypt(I, rounds=2, dim=1):
    if I is None:
        raise ValueError("You must enter at least the image")
    if rounds is None:
        rounds = 2
        warnings.warn("Rounds not found, we will use 2 rounds")
    rounds = int(np.ceil(rounds))
    if rounds < 1:
        raise ValueError("Rounds is less than 1")

    I = I.astype(np.uint8)
    M, N = I.shape

    # Ensure image size is even
    if M % 2 == 1:
        I = np.vstack((I, np.zeros((1, N), dtype=np.uint8)))
        M += 1
    if N % 2 == 1:
        I = np.hstack((I, np.zeros((M, 1), dtype=np.uint8)))
        N += 1

    MN = M * N
    SS = []

    for round_iter in range(rounds):
        # Step 2: Flatten and convert to double
        P = I.flatten().astype(np.float64)

        # Step 3: Generate chaotic sequence using logistic map
        y0 = 0.5
        x0 = (np.sum(P) + MN) / (MN + 2**23) # Initial condition from image
        
        #---------
        seq = np.zeros(MN)
        
        if dim == 1:
            x = x0 % 1
            for i in range(MN):
                x = logistic_map(x)
                seq[i] = x
        elif dim == 2:
            x, y = x0, y0
            for i in range(MN):
                x, y = henon_map(x, y)
                seq[i] = (x + 1) / 2  # normalize to [0,1]

        # Step 4: Use chaotic sequence to create a permutation
        S = np.argsort(seq)
        SS.append(S)

        # Step 5: Permute pixels
        R = P[S]

        # Step 6: Reshape and apply 2x2 matrix multiplication
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


#Decrypt method
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



# Read the image
I = cv2.imread("files/testing_color.png",0)
M, N = I.shape

# Ensure M and N are even
if M % 2 == 1:
    M += 1
if N % 2 == 1:
    N += 1

# Resize the image to the adjusted dimensions
I = cv2.resize(I, (N, M))

# Display the original image
plt.subplot(131)
plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

rounds = 2
I_enc = np.zeros_like(I)

# Encrypt each channel
SX = {}
I_enc, SX = Encrypt(I, rounds)

# Display the encrypted image
plt.subplot(132)
plt.imshow(cv2.cvtColor(I_enc, cv2.COLOR_BGR2RGB))
plt.title('Encrypted Image')


# Decrypt each channel
I_dec = np.zeros_like(I_enc)

I_dec = Decrypt(I_enc, SX)

# Display the decrypted image
plt.subplot(133)
plt.imshow(cv2.cvtColor(I_dec, cv2.COLOR_BGR2RGB))
plt.title('Decrypted Image')

plt.show()



# # Calculate MSE and PSNR
# y1 = I.flatten()
# y2 = I_dec.flatten()
# MSE = np.sum((y1 - y2) ** 2) / len(y1)

# impsnr = psnr(I_dec, I)

# # Show MSE and PSNR values
# print(f'MSE: {MSE}')
# print(f'PSNR: {impsnr}')

# plt.show()
