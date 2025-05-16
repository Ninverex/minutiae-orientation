import math
try:
    from skimage import io
except ImportError:
    exit()
try:
    import matplotlib.pyplot as plt
except ImportError:
    exit()

def image_to_list(np_array_img):
    if hasattr(np_array_img, 'tolist'):
        return np_array_img.tolist()
    if isinstance(np_array_img, list):
        return np_array_img
    raise TypeError

def rgb_to_gray_manual(rgb_image_list):
    h = len(rgb_image_list)
    if h == 0: return []
    w = len(rgb_image_list[0])
    if w == 0: return [[] for _ in range(h)]
    gray_image = [[0.0 for _ in range(w)] for _ in range(h)]
    for r in range(h):
        for c in range(w):
            pixel = rgb_image_list[r][c]
            gray_image[r][c] = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]
    return gray_image

def img_as_float_manual(image_list, current_max_val):
    if not image_list: return []
    h = len(image_list)
    if h == 0: return []
    w = len(image_list[0])
    if w == 0: return [[] for _ in range(h)]
    float_image = [[0.0 for _ in range(w)] for _ in range(h)]
    if current_max_val == 0:
        current_max_val = 1.0
    for r in range(h):
        for c in range(w):
            float_image[r][c] = image_list[r][c] / current_max_val
    return float_image

def convolve2d_manual(image, kernel):
    img_h = len(image)
    if img_h == 0: return []
    img_w = len(image[0])
    if img_w == 0: return [[] for _ in range(img_h)]
    ker_h = len(kernel)
    ker_w = len(kernel[0])
    pad_h = ker_h // 2
    pad_w = ker_w // 2
    output = [[0.0 for _ in range(img_w)] for _ in range(img_h)]
    for r_img in range(img_h):
        for c_img in range(img_w):
            sum_val = 0.0
            for r_ker in range(ker_h):
                for c_ker in range(ker_w):
                    current_r = r_img - pad_h + r_ker
                    current_c = c_img - pad_w + c_ker
                    if 0 <= current_r < img_h and 0 <= current_c < img_w:
                        sum_val += image[current_r][current_c] * kernel[r_ker][c_ker]
            output[r_img][c_img] = sum_val
    return output

def manual_sobel_filters(image):
    sobel_x_kernel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel_y_kernel = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    gx = convolve2d_manual(image, sobel_x_kernel)
    gy = convolve2d_manual(image, sobel_y_kernel)
    return gx, gy

def generate_gaussian_kernel(size=3, sigma=1.0):
    if size % 2 == 0:
        raise ValueError
    kernel = [[0.0 for _ in range(size)] for _ in range(size)]
    center = size // 2
    sum_val = 0.0
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            val = (1 / (2 * math.pi * sigma**2)) * math.exp(-(x**2 + y**2) / (2 * sigma**2))
            kernel[i][j] = val
            sum_val += val
    if sum_val == 0: sum_val = 1.0
    for i in range(size):
        for j in range(size):
            kernel[i][j] /= sum_val
    return kernel

def gaussian_blur_manual(image, sigma=1.0, kernel_size=3):
    gaussian_kernel = generate_gaussian_kernel(size=kernel_size, sigma=sigma)
    return convolve2d_manual(image, gaussian_kernel)

def compute_orientation_manual(image_list, block_size_val):
    h = len(image_list)
    if h == 0: return []
    w = len(image_list[0])
    if w == 0: return [[] for _ in range(h // block_size_val if block_size_val > 0 else 0)]
    gx_abs, gy_abs = manual_sobel_filters(image_list)
    orientation_pixels = [[math.atan2(gx_abs[r][c], gy_abs[r][c] + 1e-9) for c in range(w)] for r in range(h)]
    num_block_rows = h // block_size_val
    num_block_cols = w // block_size_val
    if num_block_rows == 0 or num_block_cols == 0: return []
    orientation_block = [[0.0 for _ in range(num_block_cols)] for _ in range(num_block_rows)]
    for i_block in range(num_block_rows):
        for j_block in range(num_block_cols):
            sum_theta = 0.0
            count_theta = 0
            start_row = i_block * block_size_val
            start_col = j_block * block_size_val
            for r_in_block in range(block_size_val):
                for c_in_block in range(block_size_val):
                    row_idx = start_row + r_in_block
                    col_idx = start_col + c_in_block
                    sum_theta += orientation_pixels[row_idx][col_idx]
                    count_theta += 1
            orientation_block[i_block][j_block] = sum_theta / count_theta if count_theta > 0 else 0.0
    return orientation_block

def show_quiver_manual(orientation_block_list, title_str, display_img_list, block_size_val):
    if not orientation_block_list:
        if display_img_list:
            plt.imshow(display_img_list, cmap='gray', vmin=0, vmax=1)
            plt.title(f"{title_str} (brak danych orientacji)")
            plt.axis('off')
        return
    block_rows = len(orientation_block_list)
    if block_rows == 0: return
    block_cols = len(orientation_block_list[0])
    if block_cols == 0: return
    X_coords, Y_coords, U_comp, V_comp = [], [], [], []
    for r_idx in range(block_rows):
        for c_idx in range(block_cols):
            X_coords.append(c_idx * block_size_val)
            Y_coords.append(r_idx * block_size_val)
            angle = orientation_block_list[r_idx][c_idx]
            U_comp.append(math.cos(angle))
            V_comp.append(math.sin(angle))
    plt.imshow(display_img_list, cmap='gray', vmin=0, vmax=1)
    V_plot = [-v for v in V_comp]
    plt.quiver(X_coords, Y_coords, U_comp, V_plot, color='red', scale=20, width=0.005, angles='uv', headwidth=3, headlength=5, pivot='tail')
    plt.title(title_str)
    plt.axis('off')

param_block_size = 16
param_gaussian_sigma = 1.0
param_gaussian_kernel_size = 3
if param_gaussian_kernel_size % 2 == 0:
    param_gaussian_kernel_size += 1

try:
    img_np = io.imread('palec.jpg')
except:
    exit()

img_ndim_np = img_np.ndim
img_dtype_str = str(img_np.dtype).lower()

if img_ndim_np == 3:
    temp_img_list_color = image_to_list(img_np)
    if img_np.shape[2] == 4:
        temp_img_list_rgb = [[pixel[:3] for pixel in row] for row in temp_img_list_color]
    elif img_np.shape[2] == 3:
        temp_img_list_rgb = temp_img_list_color
    else:
        exit()
    img_gray_list_intermediate = rgb_to_gray_manual(temp_img_list_rgb)
elif img_ndim_np == 2:
    img_gray_list_intermediate = image_to_list(img_np)
else:
    exit()

max_val_for_scaling = 1.0
is_already_float_0_1 = False

if 'int' in img_dtype_str:
    if img_dtype_str == 'uint8': max_val_for_scaling = 255.0
    elif img_dtype_str == 'uint16': max_val_for_scaling = 65535.0
    else: max_val_for_scaling = 255.0
elif 'float' in img_dtype_str:
    is_0_1_range = True
    for r_idx in range(min(len(img_gray_list_intermediate), 5)):
        for c_idx in range(min(len(img_gray_list_intermediate[0]), 5)):
            val = img_gray_list_intermediate[r_idx][c_idx]
            if not (0.0 <= val <= 1.0):
                is_0_1_range = False
                break
        if not is_0_1_range: break
    if is_0_1_range:
        is_already_float_0_1 = True
    else:
        max_val_for_scaling = 255.0

if is_already_float_0_1:
    img_float_list = img_gray_list_intermediate
else:
    img_float_list = img_as_float_manual(img_gray_list_intermediate, current_max_val=max_val_for_scaling)

if not img_float_list or (len(img_float_list) > 0 and not img_float_list[0]):
    exit()

orientation_raw_manual = compute_orientation_manual(img_float_list, param_block_size)
img_blur_manual = gaussian_blur_manual(img_float_list, sigma=param_gaussian_sigma, kernel_size=param_gaussian_kernel_size)
orientation_blur_manual = compute_orientation_manual(img_blur_manual, param_block_size)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
show_quiver_manual(orientation_raw_manual, "Orientacja PRZED filtracją (manualna)", img_float_list, param_block_size)
plt.subplot(1, 2, 2)
show_quiver_manual(orientation_blur_manual, "Orientacja PO filtracji Gauss (manualna)", img_float_list, param_block_size)
plt.tight_layout()
plt.show()
