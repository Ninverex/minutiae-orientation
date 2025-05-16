import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Parametry
block_size = 16

def load_image(filepath):
    img = Image.open(filepath)
    img = img.convert('L')  # Konwersja do skali szarości
    img = np.array(img) / 255.0  # Normalizacja do zakresu [0,1]
    return img


def custom_sobel_h(image):
    kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    return convolve2d(image, kernel)


def custom_sobel_v(image):
    kernel = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
    return convolve2d(image, kernel)


def convolve2d(image, kernel):
    k_h, k_w = kernel.shape
    i_h, i_w = image.shape

    pad_h = k_h // 2
    pad_w = k_w // 2
    output = np.zeros_like(image)

    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    # Konwolucja
    for i in range(i_h):
        for j in range(i_w):
            output[i, j] = np.sum(padded_img[i:i + k_h, j:j + k_w] * kernel)

    return output


def gaussian_filter(image, sigma=1):
    # Rozmiar jądra (3-sigma rule)
    size = int(6 * sigma + 1)
    if size % 2 == 0:
        size += 1

    # Środek jądra
    center = size // 2

    # Tworzenie siatki współrzędnych
    x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)

    # Funkcja Gaussa
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / np.sum(kernel)  # Normalizacja

    return convolve2d(image, kernel)


def compute_orientation(image, block_size=16):
    # gradienty Sobela
    gx = custom_sobel_h(image)
    gy = custom_sobel_v(image)

    # Orientacja pikseli
    orientation = np.arctan2(gy, gx)

    h, w = image.shape
    # Uśrednianie w blokach
    orientation_block = np.zeros((h // block_size, w // block_size))
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block_theta = orientation[i:i + block_size, j:j + block_size]
            orientation_block[i // block_size, j // block_size] = np.mean(block_theta)

    return orientation_block


def show_quiver(orientation_block, title, img, block_size=16):
    Y, X = np.mgrid[0:orientation_block.shape[0], 0:orientation_block.shape[1]]
    U = np.cos(orientation_block)
    V = np.sin(orientation_block)

    plt.imshow(img, cmap='gray')
    plt.quiver(X * block_size, Y * block_size, U, -V, color='red', scale=20, width=0.005)
    plt.title(title)
    plt.axis('off')




img = load_image('palec.jpg')
h, w = img.shape

# Orientacja przed filtracją
orientation_raw = compute_orientation(img)

# Orientacja po filtracji
img_blur = gaussian_filter(img, sigma=1)
orientation_blur = compute_orientation(img_blur)

# Wizualizacja
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
show_quiver(orientation_raw, "Orientacja PRZED filtracją", img)

plt.subplot(1, 2, 2)
show_quiver(orientation_blur, "Orientacja PO filtracji (Gauss)", img)

plt.tight_layout()
plt.show()