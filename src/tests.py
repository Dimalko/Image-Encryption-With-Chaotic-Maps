import numpy as np
import random

class Tests:
    
    def calculate_entropy(image):

        pixels = image.flatten()
        
        # Calculate histogram
        # Using 256 bins for grayscale images (0-255)
        histogram, _ = np.histogram(pixels, bins=256, range=(0, 255))
        
        # Normalize the histogram to get probabilities
        prob = histogram / np.sum(histogram)
        
        # Filter out zero probabilities to avoid log(0)
        prob_nonzero = prob[prob > 0]
        
        # Calculate entropy
        entropy = -np.sum(prob_nonzero * np.log2(prob_nonzero))
        
        return entropy


    def correlation_coefficient(img, num_pairs=40000):
        H, W = img.shape
        img = img.astype(np.float64)

        def calc_corr(x, y):
            ex = np.mean(x)
            ey = np.mean(y)
            dx = np.mean((x - ex) ** 2)
            dy = np.mean((y - ey) ** 2)
            cov = np.mean((x - ex) * (y - ey))
            return cov / np.sqrt(dx * dy + 1e-10) 

        directions = {'Horizontal': (0, 1), 'Vertical': (1, 0), 'Diagonal': (1, 1)}
        results = {}

        for dir_name, (di, dj) in directions.items():
            x_vals = []
            y_vals = []

            for _ in range(num_pairs):
                i = random.randint(0, H - 1 - di)
                j = random.randint(0, W - 1 - dj)
                x_vals.append(img[i, j])
                y_vals.append(img[i + di, j + dj])

            corr = calc_corr(np.array(x_vals), np.array(y_vals))
            results[dir_name] = corr

        return results
    

    def differential_attack(C1, C2):
        assert C1.shape == C2.shape
        C1 = C1.astype(np.uint8)
        C2 = C2.astype(np.uint8)
        
        M, N = C1.shape
        
        
        # NPCR calculation
        D = C1 != C2
        NPCR = np.sum(D) / (M * N) * 100

        # UACI calculation
        # !NOTE to check results,
        UACI = np.sum(np.abs(C1.astype(np.int16) - C2.astype(np.int16))) / (255 * M * N) * 100

        return NPCR, UACI
    

    def plot_histogram_with_chi2(image, title, ax):
        # Flatten image
        flat = image.flatten()
        
        # Compute histogram (exact count, no density)
        hist, _ = np.histogram(flat, bins=256, range=(0, 256))
        
        # Total pixels and expected frequency
        total_pixels = image.size
        expected = total_pixels / 256

        # Compute chi-square value
        chi2 = np.sum(((hist - expected) ** 2) / expected)

        ax.bar(np.arange(256), hist, width=1.0, color='gray')
        ax.axhline(expected, color='red', linestyle='--', linewidth=1, label='Expected')
        ax.set_title(f"{title}\nChi² = {chi2:.2f}")
        ax.set_xlim([0, 255])
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Frequency")
        ax.legend()

        return chi2


    def psnr(Io, Id):
        Io = Io.astype(np.float64)
        Id = Id.astype(np.float64)

        mse = np.mean((Io - Id) ** 2)
        if mse == 0:
            return float('inf')
        
        psnr_value = 10 * np.log10((255.0 ** 2) / mse)

        return psnr_value