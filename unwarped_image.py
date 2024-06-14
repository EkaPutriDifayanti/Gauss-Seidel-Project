import numpy as np  # Import library numpy untuk operasi matriks dan array
import cv2  # Import library OpenCV untuk membaca dan menulis gambar dalam format tertentu
import glob  # Import library glob untuk mencari file dalam folder berdasarkan pola tertentu
import os  # Import library os untuk operasi sistem seperti membuat folder dan path file sistem

# Fungsi untuk menghitung nilai rata-rata dua gambar (image1 dan image2) untuk digunakan dalam algoritma NPREG
# NPREG = Nonparametric Regression adalah metode statistik yang digunakan untuk menemukan
#         hubungan antara dua variabel tanpa mengasumsikan bentuk fungsi yang spesifik

def NPREG(image1, image2):  # Fungsi NPREG dengan parameter image1 dan image2
    # Mengembalikan nilai rata-rata dari image1 dan image2 dengan bobot 0.5 untuk masing-masing gambar
    return 0.5 * image1 + 0.5 * image2


# Fungsi untuk mengembalikan gambar yang telah diunwarp berdasarkan algoritma NPREG sebanyak N kali
def unwarp_images(I, N):
    # Menghitung jumlah gambar dalam list I dan menyimpannya dalam variabel Z
    Z = len(I)
    # Menambahkan gambar pertama dan terakhir dari list I ke dalam list I_expanded
    I_expanded = [I[0]] + I + [I[-1]]

    # Iterasi sebanyak N kali untuk mengunwarp gambar
    for t in range(N):
        # Inisialisasi variabel u dengan nilai array nol dengan ukuran yang sama dengan gambar dalam I_expanded
        u = [np.zeros_like(img) for img in I_expanded]
        # Inisialisasi variabel I_hat dengan nilai array nol dengan ukuran yang sama dengan gambar dalam I_expanded
        I_hat = [np.zeros_like(img) for img in I_expanded]
        # Inisialisasi variabel u_accu dengan nilai array nol dengan ukuran yang sama dengan gambar dalam I_expanded
        u_accu = [np.zeros_like(img) for img in I_expanded]

        for j in range(1, Z + 1):
            # Menggunakan metode Gauss-Seidel untuk menghitung nilai u[j] baru berdasarkan nilai u[j-1] dan u[j+1]
            u[j] = NPREG(I_expanded[j - 1], I_expanded[j + 1])
            # Normalisasi nilai u[j] untuk memastikan berada dalam rentang piksel yang valid
            I_hat[j] = I_expanded[j] * (1 / 2 - u[j] / 255.0)
            # Menghitung nilai u_star dengan memanfaatkan hasil sebelumnya
            u_star = NPREG(I_hat[j], I_expanded[j])
            # Menghitung nilai u_accu untuk digunakan pada iterasi berikutnya
            u_accu[j] = u[j] + u_star
            # Menggunakan nilai u_accu[j] untuk memperbarui nilai piksel pada gambar
            I_expanded[j] = I[j - 1] * (u_accu[j] / 255.0)

        # Memperbarui nilai I_expanded untuk iterasi berikutnya
        I_expanded = [I_expanded[0]] + \
            I_expanded[1:-1][::-1] + [I_expanded[-1]]

    return I_expanded[1:-1]  # Mengembalikan gambar yang telah diunwarp

# Fungsi untuk memuat gambar dari folder yang diberikan dan mengembalikan list gambar yang telah dimuat dalam skala keabuan
def load_images_from_folder(folder):
    images = []
    # Mencocokkan semua file dengan ekstensi .png dalam folder
    for filename in sorted(glob.glob(os.path.join(folder, '*.png'))):
        # Debug statement untuk melihat file yang sedang dimuat
        print(f"Loading {filename}")
        # Membaca gambar dalam skala keabuan
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        # Menambahkan gambar ke dalam list images jika berhasil dimuat
        if img is not None:  # Jika gambar berhasil dimuat
            images.append(img)  # Menambahkan gambar ke dalam list images
        else:  # Jika gambar gagal dimuat
            # Debug statement untuk melihat file yang gagal dimuat
            print(f"Failed to load {filename}")
    return images  # Mengembalikan list gambar yang telah dimuat

# Fungsi untuk menyimpan gambar ke dalam folder yang diberikan
def save_images_to_folder(images, folder):
    if not os.path.exists(folder):  # Jika folder belum ada
        os.makedirs(folder)  # Membuat folder baru
    for i, img in enumerate(images):  # Iterasi untuk setiap gambar dalam list images
        # Path untuk menyimpan gambar
        output_path = os.path.join(folder, f"unwarped_{i + 1}.png")
        # Menyimpan gambar ke dalam path yang telah ditentukan
        cv2.imwrite(output_path, img)
        # Debug statement untuk melihat file yang telah disimpan
        print(f"Saved {output_path}")


# Path ke folder yang berisi gambar input
input_folder = './input_folder'  # Path ke folder yang berisi gambar input
output_folder = './output_folder'  # Path ke folder untuk menyimpan gambar output
N = 10  # Jumlah iterasi

# Memuat gambar dari folder input
I = load_images_from_folder(input_folder)
if not I:  # Jika tidak ada gambar yang dimuat
    raise ValueError(  # Mengembalikan error jika tidak ada gambar yang dimuat
        # Debug statement untuk mengecek folder input dan format gambar
        "No images loaded. Please check the input folder and image format.")
else:  # Jika ada gambar yang dimuat
    # Debug statement untuk melihat jumlah gambar yang berhasil dimuat
    print(f"{len(I)} images loaded successfully.")

# Mengunwarp gambar dengan algoritma NPREG sebanyak N kali dan menyimpannya dalam variabel unwarped_images
unwarped_images = unwarp_images(I, N)

# Menyimpan gambar yang telah diunwarp ke dalam folder output
save_images_to_folder(unwarped_images,output_folder)