#include <WiFi.h>
#include <WiFiUdp.h>
#include <ESP32Servo.h>

// Credenciales Wi-Fi
const char* ssid = "WIFI-ALEX";
const char* password = "TU_PASSWORD";
const unsigned int localPort = 5005;

WiFiUDP udp;

// Definición de Hardware
#define PIN_SERVO      18
#define PIN_LED_VERDE  2    // Estado: Creador Autenticado / Tracking Activo
#define PIN_LED_ROJO   4    // Estado: No Reconocido / Intruso / Fallo

Servo trackerServo;
const int SERVO_CENTER = 90;
const float Kp = 0.12;       // Mapeo Proporcional (píxeles a grados)

// Estructura de Red Binaria
struct __attribute__((__packed__)) VisionPacket {
    uint32_t sequence_id;
    int16_t  error_x;
    uint8_t  status;
};

unsigned long lastPacketTime = 0;
const unsigned long WATCHDOG_TIMEOUT_MS = 300; // Timeout de Seguridad

void setup() {
    Serial.begin(115200);

    pinMode(PIN_LED_VERDE, OUTPUT);
    pinMode(PIN_LED_ROJO, OUTPUT);
   
    // Estado Inicial: Bloqueado
    digitalWrite(PIN_LED_VERDE, LOW);
    digitalWrite(PIN_LED_ROJO, HIGH);

    trackerServo.attach(PIN_SERVO);
    trackerServo.write(SERVO_CENTER);

    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(200);
        Serial.print(".");
    }
   
    udp.begin(localPort);
    Serial.println("\n[ESP32] Receptor Biométrico UDP Activo");
}

void loop() {
    int packetSize = udp.parsePacket();

    if (packetSize >= sizeof(VisionPacket)) {
        VisionPacket packet;
        udp.read((char*)&packet, sizeof(VisionPacket));

        lastPacketTime = millis(); // Reset de Watchdog

        if (packet.status == 1) {
            // ACCESO CONCEDIDO: Creador Reconocido
            digitalWrite(PIN_LED_VERDE, HIGH);
            digitalWrite(PIN_LED_ROJO, LOW);

            // Control Proporcional para seguimiento angular
            int targetAngle = SERVO_CENTER + (int)(packet.error_x * Kp);
            targetAngle = constrain(targetAngle, 20, 160); // Limite mecánico de protección
           
            trackerServo.write(targetAngle);
        }
        else {
            // ACCESO DENEGADO: Rostro No Reconocido o Ausente
            digitalWrite(PIN_LED_VERDE, LOW);
            digitalWrite(PIN_LED_ROJO, HIGH);
           
            trackerServo.write(SERVO_CENTER); // Retorno a posición de reposo
        }
    }

    // --- WATCHDOG SYSTEM (Perda de comunicación con PC) ---
    if (millis() - lastPacketTime > WATCHDOG_TIMEOUT_MS) {
        digitalWrite(PIN_LED_VERDE, LOW);
        digitalWrite(PIN_LED_ROJO, HIGH);
        trackerServo.write(SERVO_CENTER);
    }
}
