import cv2
import numpy as np
from pydicom import dcmread


def get_dicom_image(dicom_file):
    """
    Extracts metadata from a DICOM file.

    Parameters
    ----------
    dicom_file : str
        Path to the DICOM file.

    Returns
    ------- 
    numpy.ndarray
        The pixel array of the DICOM image.
    """
    try:
        ds = dcmread(dicom_file)
    except Exception as e:
        print(f"Error reading DICOM file: {e}")

    return ds.pixel_array


def apply_CLAHE_rgb(img):
    """
    Apply CLAHE to each channel of an RGB image.
    """
    b, g, r = cv2.split(img)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    cl_b = clahe.apply(b)
    cl_g = clahe.apply(g)
    cl_r = clahe.apply(r)

    cl_img = cv2.merge((cl_b, cl_g, cl_r))

    return cl_img


def find_fundus_circle(img):
    h, w = img.shape[:2]
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    img = cv2.dilate(img, kernel, iterations=2)
    img = cv2.erode(img, kernel, iterations=2)

    img = cv2.GaussianBlur(img, (9, 9), 0)
    img = cv2.Sobel(img, cv2.CV_8U, 1, 1, ksize=5)
    
    circles = cv2.HoughCircles(
        img, cv2.HOUGH_GRADIENT, dp=1, minDist=256, param1=50, param2=30, 
        minRadius=64, maxRadius=256
    )

    if circles is None:
        print("No circles found in the image.")
        return w // 2, h // 2, min(w, h) // 2

    circles = np.uint16(np.around(circles))
    x, y, r = circles[0][0]  # Get the first detected circle

    resize_factor = min(w, h) / 256
    x = int(x * resize_factor)
    y = int(y * resize_factor)
    r = int(r * resize_factor)
    
    return x, y, r


def crop_fundus_image(img, x, y, r):
    h, w = img.shape[:2]
    x1, x2 = x - r, x + r
    y1, y2 = y - r, y + r

    pad_left = max(0, -x1)
    pad_right = max(0, x2 - w)
    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - h)
    if any([pad_left, pad_right, pad_top, pad_bottom]):
        img = cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0
        )

    x1 = max(0, x1 + pad_left)
    x2 = min(img.shape[1], x2 + pad_left)
    y1 = max(0, y1 + pad_top)
    y2 = min(img.shape[0], y2 + pad_top)

    return img[y1:y2, x1:x2]


def crop_image(image, margin=0.1):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]

    x, y = np.where(binary > 0)
    
    x0, x1 = x.min(), x.max()
    y0, y1 = y.min(), y.max()
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2

    dx, dy = (x1 - x0) // 2, (y1 - y0) // 2
    d = max(dx, dy)
    d = int(d * (1 + margin))  
    x0, x1 = cx - d, cx + d
    y0, y1 = cy - d, cy + d
    bbox = (x0, x1, y0, y1)

    return image[x0:x1, y0:y1], bbox



if __name__ == "__main__":
    # Example usage
    dicom_path = "data/fundus-images/1.2.826.0.2.139953.1.2.51872.44012.57444.5.dcm"
    image = get_dicom_image(dicom_path)
    print(image.shape)

    import matplotlib.pyplot as plt
    # image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original DICOM Image')
    axs[1].imshow(apply_CLAHE_rgb(image), cmap='gray')
    axs[1].set_title('CLAHE Applied Image')
    plt.show()

