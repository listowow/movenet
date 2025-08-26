import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import csv
import time

# --- Setup Awal ---
# Muat model Movenet SinglePose Lightning dari TensorFlow Hub
# Lightning lebih cepat dan cocok untuk deteksi real-time.
print("Memuat model Movenet. Tunggu sebentar...")
model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
model = hub.load(model_url)
movenet = model.signatures['serving_default']
print("Model berhasil dimuat.")

# Definisi keypoints dan pasangannya untuk visualisasi
# Ini digunakan untuk menggambar garis antar keypoints, seperti bahu ke siku.
KEYPOINT_DICT = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}
CONNECTIONS = [
    (KEYPOINT_DICT['left_shoulder'], KEYPOINT_DICT['right_shoulder']),
    (KEYPOINT_DICT['left_shoulder'], KEYPOINT_DICT['left_elbow']),
    (KEYPOINT_DICT['right_shoulder'], KEYPOINT_DICT['right_elbow']),
    (KEYPOINT_DICT['left_elbow'], KEYPOINT_DICT['left_wrist']),
    (KEYPOINT_DICT['right_elbow'], KEYPOINT_DICT['right_wrist']),
    (KEYPOINT_DICT['left_shoulder'], KEYPOINT_DICT['left_hip']),
    (KEYPOINT_DICT['right_shoulder'], KEYPOINT_DICT['right_hip']),
    (KEYPOINT_DICT['left_hip'], KEYPOINT_DICT['right_hip']),
    (KEYPOINT_DICT['left_hip'], KEYPOINT_DICT['left_knee']),
    (KEYPOINT_DICT['right_hip'], KEYPOINT_DICT['right_knee']),
    (KEYPOINT_DICT['left_knee'], KEYPOINT_DICT['left_ankle']),
    (KEYPOINT_DICT['right_knee'], KEYPOINT_DICT['right_ankle'])
]

# Fungsi untuk menggambar keypoints dan koneksi
def draw_keypoints_and_connections(frame, keypoints, confidence_threshold=0.3):
    height, width, _ = frame.shape
    
    # Gambar koneksi (garis)
    for p1, p2 in CONNECTIONS:
        y1, x1, score1 = keypoints[p1]
        y2, x2, score2 = keypoints[p2]
        if score1 > confidence_threshold and score2 > confidence_threshold:
            start_point = (int(x1 * width), int(y1 * height))
            end_point = (int(x2 * width), int(y2 * height))
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

    # Gambar keypoints (titik)
    for y, x, score in keypoints:
        if score > confidence_threshold:
            center = (int(x * width), int(y * height))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

# Fungsi untuk memproses frame video
def run_inference(image):
    # Ubah ukuran gambar menjadi 192x192, ukuran yang dibutuhkan Movenet
    input_tensor = tf.image.resize_with_pad(image, 192, 192)
    # Konversi ke tipe data yang diperlukan model (int32)
    # Perbaikan: Mengubah dtype dari float32 ke int32
    input_tensor = tf.cast(input_tensor, dtype=tf.int32)
    
    # Tambahkan dimensi batch
    input_tensor = tf.expand_dims(input_tensor, axis=0)
    
    # Jalankan inferensi
    outputs = movenet(input_tensor)
    
    # Ambil output: [1, 1, 17, 3] -> (batch, person, keypoint, [y, x, score])
    keypoints_with_scores = outputs['output_0'].numpy()
    
    return keypoints_with_scores[0][0]

# --- Main Program ---
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak bisa membuka kamera.")
        return

    is_recording = False
    recorded_data = []

    # Buat header untuk file CSV
    # Nama-nama kolom akan berisi 'timestamp' dan 'nama_keypoint_x/y/score'
    header = ['timestamp']
    for keypoint_name in KEYPOINT_DICT:
        header.append(f'{keypoint_name}_y')
        header.append(f'{keypoint_name}_x')
        header.append(f'{keypoint_name}_score')
    
    print("Tekan 'R' untuk mulai/berhenti merekam.")
    print("Tekan 'Q' untuk keluar dari program.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Balik frame agar seperti cermin
        frame = cv2.flip(frame, 1)

        # Jalankan inferensi Movenet pada frame
        keypoints = run_inference(frame)

        # Jika sedang merekam, simpan data keypoint
        if is_recording:
            timestamp = time.time()
            row = [timestamp]
            for kp in keypoints:
                row.extend(kp)
            recorded_data.append(row)
            cv2.putText(frame, "REKAMAN...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Gambar keypoints dan koneksi pada frame
        draw_keypoints_and_connections(frame, keypoints)

        # Tampilkan frame
        cv2.imshow('Movenet Webcam', frame)

        # Tangkap tombol yang ditekan
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            is_recording = not is_recording
            if is_recording:
                print("Merekam dimulai...")
            else:
                print("Merekam dihentikan.")

    # Setelah loop selesai, simpan data yang direkam ke file CSV
    if recorded_data:
        csv_file_name = f'movenet_data_{int(time.time())}.csv'
        with open(csv_file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(recorded_data)
        print(f"Data pose telah disimpan ke '{csv_file_name}'")
    
    # Bersihkan semua jendela dan lepaskan kamera
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
