import cv2
import math
import numpy as np
from scipy.integrate import solve_ivp


#display image
def disp_img(n, i):
    cv2.imshow(n, i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



#generate a keystream using a logistic map
def logistic_map(x, a):
    return a * x * (1 - x)

def tent_map(x, a):
    if x > 0.5:
        return (a * x) % 1 #keeps x within 0 and 1
    else:
        return (a * (1 - x)) % 1
    
def sine_map(x, a):
    return a * math.sin(math.pi * x)

def cubic_map(x, a):
    return a * x * (1 - x**2)

def cosine_polynomial_map(x, a):
    return math.cos(a * (x**3 + x))

def henon_map(x, y, a, b):
    x2 = 1 - a * x**2 + y
    y2 = b * x

    x2 = max(-1, min(1, x2))  #keeps x between -1 and 1 to prevent overflow error
    return x2, y2

def gingerbread_man_map(x, y):
    x2 = 1 - y + abs(x)
    y2 = x
    return x2, y2

def lozi_map(x, y, a, b):
    x2 = 1 + y - a * abs(x)
    y2 = b * x
    return x2, y2
    


#load image
img = cv2.imread("files/testing_color.png",0)

#convert image to array format
im_array = np.array(img)
rows, cols = im_array.shape

#choose dimentions of the map
dimentions = 2 #accepted values: 1 or 2 only

#initialize parameters
x = 0.002
y = 0.3
l = 1.4 #parameter 1
m = 0.3 #parameter 2
chaos = 3064

#encrypted machine
if dimentions == 1:
    keystream = []
    keystream_values = []
    
    for i in range(im_array.size):
        x = tent_map(x, l)
        keystream_values.append(x) #store for graph
        _x = int(x * chaos) %256
        keystream.append(_x)
        #print(f"Iteration {i} x={x}")

elif dimentions == 2:
    keystream = np.zeros((rows, cols), dtype=np.uint8)
    keystream_values = []
    
    for i in range(rows):
        for j in range(cols):
            x, y = lozi_map(x, y, l, m)
            keystream_values.append(x) #store for graph
            _xy = int((x + y) * chaos) %256
            keystream[i, j] = _xy
            #print(f"Iteration {i}, {j}: x={x}, y={y}")


#convert keystream to same shape as image
keystream_array = np.array(keystream, dtype=np.uint8).reshape(rows, cols)

#display image
disp_img("Grayscaled Image", img)

#encryption: apply XOR operation
enc_img = np.bitwise_xor(im_array, keystream_array)
#display encrypted image
disp_img("Encrypted Image", enc_img)

#decryption: apply XOR operation again
dec_img = np.bitwise_xor(enc_img, keystream_array)
#display decrypted image
disp_img("Decrypted Image", dec_img)


def encrypt(I, rounds=2):
    I = I.astype(np.uint8)
    M, N = I.shape
    MN = M * N
    SS = []

    for round_iter in range(rounds):
        # Step 2: Flatten and convert to double
        P = I.flatten().astype(np.float64)

        # Step 3: Initial x array
        x = [(np.sum(P) + MN) / (MN + (2**23))]
        for i in range(2, 7):
            x.append(np.mod(x[i-1] * 1e6, 1))

        # Step 4: Define chaotic system
        def L(t, x):
            a, b, c, d, e, r = 10, 8/3, 28, -1, 8, 3
            return [
                a * (x[1] - x[0]) + x[3] - x[4] - x[5],
                c * x[0] - x[1] - x[0] * x[2],
                -b * x[2] + x[0] * x[1],
                d * x[3] - x[1] * x[2],
                e * x[5] + x[2] * x[1],
                r * x[0]
            ]

        N0 = 0.9865 * MN / 3
        MN3 = int(np.ceil(MN / 3))
        sol = solve_ivp(L, [N0, MN3], x, t_eval=np.linspace(N0, MN3, MN3))
        Y = sol.y.T

        # Step 5: Prepare L and get permutation S
        L_vals = Y[:MN3, [0, 2, 4]].flatten()[:MN]
        S = np.argsort(L_vals)
        SS.append(S)

        # Step 6: Apply permutation
        R = P[S]

        # Step 7: Reshape and perform matrix multiplication
        R_ = R.reshape(M, N)
        A = np.array([[89, 55], [55, 34]])
        C = np.zeros((M, N), dtype=np.float64)

        for i in range(0, M, 2):
            for j in range(0, N, 2):
                Cx = np.array([[R_[i, j], R_[i, j+1]],
                               [R_[i+1, j], R_[i+1, j+1]]])
                fz = np.dot(Cx, A)
                C[i, j], C[i, j+1] = fz[0, 0], fz[0, 1]
                C[i+1, j], C[i+1, j+1] = fz[1, 0], fz[1, 1]

        I = np.mod(C, 256).astype(np.uint8)

    return I, SS