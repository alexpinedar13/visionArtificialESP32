import cv2
import numpy as np
import socket
import struct
import os


# ============================================================
# CONFIGURACIÓN
# ============================================================

ESP32_IP = "192.168.1.150"
ESP32_PORT = 5005

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

FRAME_CENTER_X = FRAME_WIDTH // 2

# LBPH:
# Un valor MENOR significa mayor similitud.
CONFIDENCE_THRESHOLD = 70.0


# ============================================================
# RUTAS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CASCADE_PATH = os.path.join(
    BASE_DIR,
    "haarcascade_frontalface_default.xml"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "modelo_creador.xml"
)


# ============================================================
# COMPROBAR ARCHIVOS
# ============================================================

if not os.path.isfile(CASCADE_PATH):
    raise FileNotFoundError(
        "\n[ERROR] No se encontró el clasificador Haar:\n"
        f"{CASCADE_PATH}\n\n"
        "Coloca 'haarcascade_frontalface_default.xml' "
        "junto a este programa."
    )


if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        "\n[ERROR] No se encontró el modelo LBPH:\n"
        f"{MODEL_PATH}\n\n"
        "Ejecuta primero el programa de entrenamiento."
    )


# ============================================================
# CARGAR CLASIFICADOR HAAR
# ============================================================

face_cascade = cv2.CascadeClassifier(
    CASCADE_PATH
)

if face_cascade.empty():
    raise RuntimeError(
        "[ERROR] El clasificador Haar no pudo cargarse."
    )

print("[INFO] Clasificador Haar cargado correctamente.")


# ============================================================
# COMPROBAR OPENCV-CONTRIB
# ============================================================

if not hasattr(cv2, "face"):
    raise RuntimeError(
        "\n[ERROR] Tu instalación de OpenCV no contiene cv2.face.\n\n"
        "Instala opencv-contrib-python:\n\n"
        "pip uninstall opencv-python opencv-contrib-python "
        "opencv-python-headless opencv-contrib-python-headless -y\n\n"
        "pip install opencv-contrib-python"
    )


# ============================================================
# CARGAR MODELO LBPH
# ============================================================

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

try:
    face_recognizer.read(MODEL_PATH)
except Exception as e:
    raise RuntimeError(
        f"[ERROR] No se pudo cargar el modelo LBPH:\n"
        f"{MODEL_PATH}\n\n"
        f"Detalle: {e}"
    )

print("[INFO] Modelo LBPH cargado correctamente.")
print(f"[INFO] Modelo: {MODEL_PATH}")


# ============================================================
# CONFIGURAR UDP
# ============================================================

sock = socket.socket(
    socket.AF_INET,
    socket.SOCK_DGRAM
)

ESP32_ADDRESS = (
    ESP32_IP,
    ESP32_PORT
)

print(
    f"[INFO] UDP configurado: "
    f"{ESP32_IP}:{ESP32_PORT}"
)


# ============================================================
# ABRIR CÁMARA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)


if not cap.isOpened():
    sock.close()

    raise RuntimeError(
        "[ERROR] No se pudo abrir la cámara."
    )


print("[INFO] Cámara iniciada.")
print("[INFO] Presiona Q para salir.")


# ============================================================
# VARIABLES
# ============================================================

sequence_id = 0


# ============================================================
# BUCLE PRINCIPAL
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # CAPTURAR FRAME
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:
            print(
                "[ERROR] No se pudo leer un frame de la cámara."
            )
            break


        # ----------------------------------------------------
        # ASEGURAR TAMAÑO
        # ----------------------------------------------------

        height, width = frame.shape[:2]

        frame_center_x = width // 2
        frame_center_y = height // 2


        # ----------------------------------------------------
        # CONVERTIR A ESCALA DE GRISES
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # ----------------------------------------------------
        # DETECTAR ROSTROS
        # ----------------------------------------------------

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(100, 100)
        )


        # ----------------------------------------------------
        # ESTADO INICIAL
        # ----------------------------------------------------

        error_x = 0

        # 0 = sin rostro / desconocido
        # 1 = creador reconocido

        status = 0

        creator_found = False


        # ----------------------------------------------------
        # PROCESAR ROSTROS
        # ----------------------------------------------------

        for (x, y, w, h) in faces:

            face_roi = gray[
                y:y+h,
                x:x+w
            ]


            # Evitar ROI inválido
            if face_roi.size == 0:
                continue


            # Redimensionar igual que durante entrenamiento
            face_roi = cv2.resize(
                face_roi,
                (150, 150)
            )


            # ------------------------------------------------
            # PREDICCIÓN LBPH
            # ------------------------------------------------

            label, confidence = face_recognizer.predict(
                face_roi
            )


            # Centro del rostro
            cx = x + (w // 2)
            cy = y + (h // 2)


            # ------------------------------------------------
            # COMPROBAR IDENTIDAD
            # ------------------------------------------------

            if (
                label == 0
                and confidence < CONFIDENCE_THRESHOLD
            ):

                # ============================================
                # CREADOR RECONOCIDO
                # ============================================

                status = 1

                error_x = cx - frame_center_x

                creator_found = True


                # Rectángulo verde
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )


                # Texto
                cv2.putText(
                    frame,
                    f"Creador ({confidence:.1f})",
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )


                # Centro del rostro
                cv2.circle(
                    frame,
                    (cx, cy),
                    5,
                    (0, 255, 0),
                    -1
                )


                # Línea desde centro de cámara
                cv2.line(
                    frame,
                    (frame_center_x, frame_center_y),
                    (cx, cy),
                    (0, 255, 0),
                    2
                )


                # Solo necesitamos un creador
                break


            else:

                # ============================================
                # DESCONOCIDO
                # ============================================

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 0, 255),
                    2
                )


                cv2.putText(
                    frame,
                    f"Desconocido ({confidence:.1f})",
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )


        # ====================================================
        # LÍNEA CENTRAL DE LA CÁMARA
        # ====================================================

        cv2.line(
            frame,
            (frame_center_x, 0),
            (frame_center_x, height),
            (255, 255, 255),
            1
        )


        # ====================================================
        # INFORMACIÓN EN PANTALLA
        # ====================================================

        if creator_found:

            status_text = "CREADOR RECONOCIDO"
            status_color = (0, 255, 0)

        elif len(faces) > 0:

            status_text = "ROSTRO DESCONOCIDO"
            status_color = (0, 0, 255)

        else:

            status_text = "SIN ROSTRO"
            status_color = (0, 255, 255)


        cv2.putText(
            frame,
            status_text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2
        )


        cv2.putText(
            frame,
            f"Error X: {error_x}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Status: {status}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # ====================================================
        # ENVIAR DATOS POR UDP
        # ====================================================

        sequence_id += 1


        # uint32 = sequence_id
        # int16  = error_x
        # int8   = status
        #
        # Little Endian
        #
        # <Ihb
        #
        # 4 bytes + 2 bytes + 1 byte = 7 bytes

        payload = struct.pack(
            "<Ihb",
            sequence_id,
            int(error_x),
            int(status)
        )


        try:

            sock.sendto(
                payload,
                ESP32_ADDRESS
            )

        except OSError as e:

            print(
                f"[WARNING] Error enviando UDP: {e}"
            )


        # ====================================================
        # MOSTRAR VIDEO
        # ====================================================

        cv2.imshow(
            "Nodo Biometrico CPS",
            frame
        )


        # ====================================================
        # SALIR CON Q
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            print("[INFO] Finalizando programa...")
            break


# ============================================================
# LIMPIEZA
# ============================================================

finally:

    print("[INFO] Liberando recursos...")

    cap.release()

    cv2.destroyAllWindows()

    sock.close()

    print("[INFO] Programa finalizado.")
