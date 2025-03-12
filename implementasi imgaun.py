import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
from PIL import Image

# Buka gambar dan ubah ke numpy array dengan tipe uint8
image = np.array(Image.open('bunny.png').convert("RGB"), dtype=np.uint8)

# Definisi augmentor
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # Flip horizontal dengan peluang 50%
    iaa.Affine(rotate=(-10, 10)),  # Rotasi gambar dalam rentang -10 sampai 10 derajat
    iaa.GaussianBlur(sigma=(0, 1.0))  # Blur Gaussian dengan sigma antara 0 dan 1.0
])

# Augmentasi gambar (harus dalam bentuk list)
augmented_image = seq(images=[image])[0]  # Ambil elemen pertama dari hasil augmentasi

# Tampilkan hasil augmentasi
plt.figure(figsize=(10, 5))

# Gambar asli
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Gambar Asli')
plt.axis("off")

# Gambar setelah augmentasi
plt.subplot(1, 2, 2)
plt.imshow(augmented_image)
plt.title('Gambar Setelah Augmentasi')
plt.axis("off")

plt.tight_layout()
plt.show()
