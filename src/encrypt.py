import numpy as np
from scipy.integrate import solve_ivp



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
