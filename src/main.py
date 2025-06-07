import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise 
from tests import Tests as t
from encrypt import Encrypt as e
from decrypt import Decrypt as d



#Usage Example
I = cv2.imread("files/boat.tiff",0)
M, N = I.shape

if M % 2 == 1:
    M += 1
if N % 2 == 1:
    N += 1

I = cv2.resize(I, (N, M))

rounds = 2
I_enc = np.zeros_like(I)

I_enc, SX = e.encrypt(I, rounds)

I_dec = np.zeros_like(I_enc)
I_dec = d.decrypt(I_enc, SX)



# Testing PSNR
#noise
I_enc_n = random_noise(I_enc, mode='s&p', amount=0.002)
I_enc_n = (I_enc_n * 255).astype(np.uint8)
I_dec_n = d.decrypt(I_enc_n, SX)
impsnr = t.psnr(I, I_dec_n)
print("\nSalt and Pepper with noise level 0.002")
print(f'PSNR: {impsnr}')

I_enc_n_2 = random_noise(I_enc, mode='s&p', amount=0.005)
I_enc_n_2 = (I_enc_n_2 * 255).astype(np.uint8)
I_dec_n_2 = d.decrypt(I_enc_n_2, SX)
impsnr = t.psnr(I, I_dec_n_2)
print("\nSalt and Pepper with noise level 0.005")
print(f'PSNR: {impsnr}')

#data cut
I_enc_c = I_enc.copy()
I_enc_c[:128, :128] = 0 
I_enc_c = (I_enc_c * 255)
I_dec_c = d.decrypt(I_enc_c, SX)
impsnr = t.psnr(I, I_dec_c)
print("\ndata cut of 128x128")
print(f'PSNR: {impsnr}')

I_enc_c_2 = I_enc.copy()
I_enc_c_2[:64, :64] = 0 
I_enc_c_2 = (I_enc_c_2 * 255)
I_dec_c_2 = d.decrypt(I_enc_c_2, SX)
impsnr = t.psnr(I, I_dec_c_2)
print("\ndata cut of 64x64")
print(f'PSNR: {impsnr}')

# Testing Entropy
entropy_original = t.calculate_entropy(I)
entropy_encrypted = t.calculate_entropy(I_enc)
entropy_decrypted = t.calculate_entropy(I_dec)

print(f"\nEntropy of original image: {entropy_original:.4f}")
print(f"Entropy of encrypted image: {entropy_encrypted:.4f}")
print(f"Entropy of decrypted image: {entropy_decrypted:.4f}")


# Testing Correlation Coefficient
results = t.correlation_coefficient(I_enc)
print("\nCorrelation Coefficients (Encrypted Image):")
for direction, value in results.items():
    print(f"{direction}: {value:.5f}")


# Testing Differential Attack
I_ch = I.copy().astype(np.float64)
I_ch[0, 0] = (I_ch[0, 0] + 1) % 256  # Change one pixel


 # Encrypt the changed by one pixel image
I_enc_ch, _ = e.encrypt(I_ch, rounds)

npcr, uaci = t.differential_attack(I_enc, I_enc_ch)

print(f"\nNPCR: {npcr:.4f}%")
print(f"UACI: {uaci:.4f}%")



# Plotting the results
fig, axs = plt.subplots(4, 3, figsize=(18, 20))  # 4 rows, 3 columns
fig.suptitle("Image Encryption Analysis", fontsize=20)

# Row 1: Original, Encrypted, Decrypted
axs[0, 0].imshow(I, cmap='gray')
axs[0, 0].set_title("Original Image")
axs[0, 0].axis("off")

axs[0, 1].imshow(I_enc, cmap='gray')
axs[0, 1].set_title("Encrypted Image")
axs[0, 1].axis("off")

axs[0, 2].imshow(I_dec, cmap='gray')
axs[0, 2].set_title("Decrypted Image")
axs[0, 2].axis("off")

# Row 2: S&P Noise Encrypted, Decrypted
axs[1, 0].imshow(I_enc_n, cmap='gray')
axs[1, 0].set_title("Encrypted + S&P Noise")
axs[1, 0].axis("off")

axs[1, 1].imshow(I_dec_n, cmap='gray')
axs[1, 1].set_title("Decrypted from Noisy")
axs[1, 1].axis("off")

axs[1, 2].axis("off")  # Empty cell

# Row 3: Data Cut Encrypted, Decrypted
axs[2, 0].imshow(I_enc_c, cmap='gray')
axs[2, 0].set_title("Encrypted with Data Cut")
axs[2, 0].axis("off")

axs[2, 1].imshow(I_dec_c, cmap='gray')
axs[2, 1].set_title("Decrypted from Data Cut")
axs[2, 1].axis("off")

axs[2, 2].axis("off")  # Empty cell

# Row 4: Histograms
_ = t.plot_histogram_with_chi2(I, "Original Histogram", axs[3, 0])
_ = t.plot_histogram_with_chi2(I_enc, "Encrypted Histogram", axs[3, 1])
_ = t.plot_histogram_with_chi2(I_dec, "Decrypted Histogram", axs[3, 2])

# Clean layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave space for suptitle
plt.show()
