// ESP32: lee el VL53L0X (I2C) y envía la distancia en mm por serial, una línea por lectura.
#include <Wire.h>
#include <Adafruit_VL53L0X.h>

Adafruit_VL53L0X tof;

void setup() {
  Serial.begin(115200);
  Wire.begin();                      // SDA/SCL por defecto del ESP32 (GPIO21/22)
  if (!tof.begin()) { while (1) delay(10); }   // sensor no encontrado
  tof.startRangeContinuous();
}

void loop() {
  if (tof.isRangeComplete()) {
    uint16_t mm = tof.readRange();   // valores muy altos = fuera de rango
    Serial.println(mm);
  }
}
