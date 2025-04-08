import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


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



# Display histograms of original, encrypted, and decrypted images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(im_array.ravel(), bins=256, color='black', alpha=0.7)
plt.title("Histogram - Original Image")

plt.subplot(1, 3, 2)
plt.hist(enc_img.ravel(), bins=256, color='red', alpha=0.7)
plt.title("Histogram - Encrypted Image")

plt.subplot(1, 3, 3)
plt.hist(dec_img.ravel(), bins=256, color='green', alpha=0.7)
plt.title("Histogram - Decrypted Image")

plt.show()