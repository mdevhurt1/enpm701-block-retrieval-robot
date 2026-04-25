/*
 * imu_serial.ino - Read BNO055 IMU heading and send over USB serial.
 *
 * Hardware:
 *   Arduino Nano  +  Adafruit BNO055 breakout
 *   BNO055 SDA -> Arduino A4 (SDA)
 *   BNO055 SCL -> Arduino A5 (SCL)
 *   BNO055 VIN -> 3.3 V or 5 V (check breakout specs)
 *   BNO055 GND -> GND
 *
 * Output (9600 baud, USB serial):
 *   One float per line, e.g.:  "123.45\n"
 *   This matches the format expected by robot.py::_imu_reader().
 *
 * Dependencies (install via Arduino Library Manager):
 *   - Adafruit BNO055
 *   - Adafruit Unified Sensor
 *   - Wire (built-in)
 */

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

// BNO055 I2C address: 0x28 (ADR pin low) or 0x29 (ADR pin high)
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);

void setup() {
    Serial.begin(115200);

    if (!bno.begin()) {
        // Sensor not found - flash the onboard LED and halt
        pinMode(LED_BUILTIN, OUTPUT);
        while (true) {
            digitalWrite(LED_BUILTIN, HIGH);
            delay(200);
            digitalWrite(LED_BUILTIN, LOW);
            delay(200);
        }
    }

    // Use external crystal for better accuracy
    bno.setExtCrystalUse(true);

    // Brief stabilisation delay
    delay(1000);
}

void loop() {
    // Get Euler angles from the BNO055
    sensors_event_t event;
    bno.getEvent(&event);

    // event.orientation.x is the heading (yaw) in degrees [0, 360)
    float heading = event.orientation.x;

    // Send heading as plain float string, one per line
    Serial.println(heading, 2);

    // ~50 Hz update rate
    delay(20);
}
