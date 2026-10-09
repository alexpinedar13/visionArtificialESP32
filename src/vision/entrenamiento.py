import cv2
import os
import numpy as np

# ============================================================
# CONFIGURACIÓN
# ============================================================

DATA_PATH = "dataset_creador"

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# ============================================================
# CARGAR CLASIFICADOR HAAR
# ============================================================

CASCADE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    raise RuntimeError(
        f"No se pudo cargar el clasificador Haar:\n{CASCADE_PATH}"
    )

print("[INFO] Clasificador Haar cargado correctamente.")

# ============================================================
# CÁMARA
# ============================================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara.")

count = 0

print("[INFO] Enfoca tu rostro a la cámara.")
print("[INFO] Capturando 50 muestras...")

# ============================================================
# CAPTURA DE ROSTROS
# ============================================================

while count < 50:

    ret, frame = cap.read()

    if not ret:
        print("[ERROR] No se pudo leer la cámara.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:

        count += 1

        face_roi = gray[y:y+h, x:x+w]

        face_roi = cv2.resize(
            face_roi,
            (150, 150)
        )

        # Guardar imagen
        image_path = os.path.join(
            DATA_PATH,
            f"rostro_{count}.jpg"
        )

        cv2.imwrite(
            image_path,
            face_roi
        )

        # Dibujar rectángulo
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Muestra: {count}/50",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # Evitar pasar de 50
        if count >= 50:
            break

    cv2.imshow(
        "Registro Biometrico del Creador",
        frame
    )

    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

# ============================================================
# LIBERAR CÁMARA
# ============================================================

cap.release()
cv2.destroyAllWindows()

# ============================================================
# ENTRENAMIENTO LBPH
# ============================================================

print("[INFO] Entrenando reconocedor LBPH...")

faces_data = []
labels = []

for file_name in os.listdir(DATA_PATH):

    if file_name.lower().endswith(".jpg"):

        image_path = os.path.join(
            DATA_PATH,
            file_name
        )

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            print(f"[WARNING] No se pudo leer: {image_path}")
            continue

        faces_data.append(image)
        labels.append(0)

# Comprobar que tenemos imágenes
if len(faces_data) == 0:
    raise RuntimeError(
        "No se encontraron imágenes para entrenar."
    )

print(f"[INFO] Imágenes encontradas: {len(faces_data)}")

# ============================================================
# CREAR RECONOCEDOR LBPH
# ============================================================

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.train(
    faces_data,
    np.array(labels)
)

face_recognizer.write(
    "modelo_creador.xml"
)

print("[EXITO] Modelo biométrico guardado en:")
print("modelo_creador.xml")
