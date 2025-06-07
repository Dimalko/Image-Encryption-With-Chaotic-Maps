import numpy as np

class Decrypt:

    def decrypt(I_enc, SS):
        
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

