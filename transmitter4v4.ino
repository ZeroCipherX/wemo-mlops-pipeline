/*  WEMO NODE v2.4 — Enterprise Energy Monitor
 *  ─────────────────────────────────────────────────────────────
 *  WiFi credential flow:
 *
 *  FIRST BOOT (NVS empty / after flash erase):
 *    → AP "WEMO-Setup" / "wemo1234" starts at 192.168.4.1
 *    → Connect phone to "WEMO-Setup", open 192.168.4.1
 *    → Enter your home WiFi SSID + password → Save
 *    → Credentials written to NVS flash → device restarts
 *    → AP NEVER appears again unless you erase flash
 *
 *  EVERY BOOT AFTER THAT (NVS has credentials):
 *    → Connects to saved WiFi directly, no AP at all
 *    → To update credentials later: dashboard WiFi Setup tab
 *      sends {"ssid":"…","pass":"…"} to wemo/wifi-config
 *      → saved to NVS → restarts → done
 *
 *  OFFLINE MODE (saved credentials but WiFi unavailable):
 *    → No AP, no hotspot — just runs offline
 *    → Button + LoRa + PZEM + TFT all work perfectly
 *    → TFT shows "Global: OFF | Btn+LoRa OK"
 *
 *  RELAY CONTROL:
 *    → Physical button: always works, zero WiFi dependency
 *    → MQTT dashboard (http://13.60.236.168): works when online
 *    → LoRa remote button: works online and offline
 *  ─────────────────────────────────────────────────────────────
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <SPI.h>
#include <LoRa.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>
#include <PZEM004Tv30.h>
#include <Preferences.h>
#include <WebServer.h>
#include <DNSServer.h>

#include <HTTPClient.h> // to connect it with flask

// ─── AP CONFIG (first-boot provisioning only) ─────────────────
const char* AP_SSID = "WEMO-Setup";
const char* AP_PASS = "wemo1234";      // min 8 chars — change if you want

// ─── MQTT CONFIG ──────────────────────────────────────────────
const char* MQTT_HOST = "13.60.236.168";
const int   MQTT_PORT = 1883;

// ─── FLASK CONFIG ─────────────────────────────────────────────
// const char* FLASK_URL = "http://13.60.236.168:5000/api/data";   // ← update port/path if needed
const char* FLASK_URL       = "http://192.168.100.32:5000/api/data";
const char* FLASK_RELAY_URL = "http://192.168.100.32:5000/api/relay-state";

// ─── SAFETY LIMITS ────────────────────────────────────────────
const float MAX_WATTS = 2300.0f;
const float MAX_AMPS  = 16.0f;
const float MAX_VOLTS = 260.0f;

// ─── PINS ─────────────────────────────────────────────────────
#define TFT_CS    15
#define TFT_DC    27
#define TFT_RST   26
#define LORA_CS    5
#define LORA_RST   4
#define LORA_DIO0  2
#define RELAY_PIN 14
#define BTN_PIN   13

// ─── RGB565 PALETTE ───────────────────────────────────────────
#define C_BG      0x0841
#define C_PANEL   0x0C63
#define C_BORDER  0x2945
#define C_BLACK   0x0000
#define C_WHITE   0xFFFF
#define C_GRAY    0x8410
#define C_MGRAY   0x4208
#define C_YELLOW  0xFFE0
#define C_CYAN    0x07FF
#define C_GREEN   0x07E0
#define C_DKGRN   0x0300
#define C_RED     0xF800
#define C_DKRED   0x6000
#define C_ORANGE  0xFD20
#define C_PURPLE  0x801F
#define C_TEAL    0x0410
#define C_BLUE    0x001F
#define C_SKYBLUE 0x3DFF

// ─── OBJECTS ──────────────────────────────────────────────────
PZEM004Tv30     pzem(Serial2, 32, 33);
Adafruit_ST7735 tft(TFT_CS, TFT_DC, TFT_RST);
WiFiClient      espClient;
PubSubClient    mqtt(espClient);
Preferences     prefs;
WebServer       httpServer(80);   // only used during first-boot AP mode
DNSServer       dnsServer;        // captive portal redirect

// ─── STATE ────────────────────────────────────────────────────
bool          relayOn    = false;
bool          mqttOK     = false;
bool          loraOK     = false;
bool          pzemOK     = false;
bool          wifiOK     = false;
bool          apMode     = false;   // true ONLY during first-boot AP setup
unsigned long lastTX     = 0;
unsigned long bootTime   = 0;
unsigned long lastBtn    = 0;
String        wifiSSID   = "";
String        wifiPass   = "";

// ─── PROTOTYPES ───────────────────────────────────────────────
void mqttCallback(char*, byte*, unsigned int);
void updateSystem();
void drawTFT(float v, float i, float p, float e, float f, float pf, const String& up);
void drawBox(int x, int y, int w, int h, uint16_t col);
void drawBar(int x, int y, int w, int h, float ratio, uint16_t col);
void drawDot(int x, int y, bool ok, uint16_t okCol = C_GREEN);
String fmtUptime(unsigned long ms);
void drawBootScreen();
void bootStatus(const char* sys, const char* msg, uint16_t col, int y);
void bootDone();
void loadCredentials();
void saveCredentials(const String& ssid, const String& pass);
bool connectWiFi(const String& ssid, const String& pass, uint16_t timeoutMs);
void startFirstBootAP();
void handleAPRoot();
void handleAPSave();
void handleAPScan();

// ══════════════════════════════════════════════════════════════
//  NVS CREDENTIAL STORAGE
// ══════════════════════════════════════════════════════════════
void loadCredentials() {
  prefs.begin("wemo-cfg", true);
  wifiSSID = prefs.getString("ssid", "");
  wifiPass = prefs.getString("pass", "");
  prefs.end();
  Serial.println(wifiSSID.length() ? "NVS SSID: " + wifiSSID : "NVS empty — first boot AP will start");
}

void saveCredentials(const String& ssid, const String& pass) {
  prefs.begin("wemo-cfg", false);
  prefs.putString("ssid", ssid);
  prefs.putString("pass", pass);
  prefs.end();
  Serial.println("Credentials saved to NVS: " + ssid);
}

// ══════════════════════════════════════════════════════════════
//  WIFI CONNECTION  —  STA only, no AP after first boot
// ══════════════════════════════════════════════════════════════
bool connectWiFi(const String& ssid, const String& pass, uint16_t timeoutMs = 15000) {
  if (ssid.length() == 0) return false;
  WiFi.mode(WIFI_AP_STA);
  WiFi.begin(ssid.c_str(), pass.c_str());
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < timeoutMs) {
    delay(300);
  }
  return (WiFi.status() == WL_CONNECTED);
}

// ══════════════════════════════════════════════════════════════
//  FIRST-BOOT AP + CAPTIVE PORTAL
//  Called ONLY when NVS is empty (wifiSSID.length() == 0).
//  Once credentials are saved and device restarts, this never
//  runs again — NVS will have the SSID from that point on.
// ══════════════════════════════════════════════════════════════
void startFirstBootAP() {
  apMode = true;
  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASS);
  delay(200);

  // Captive-portal DNS — redirect every hostname to our IP
  dnsServer.start(53, "*", WiFi.softAPIP());

  httpServer.on("/",                          HTTP_GET,  handleAPRoot);
  httpServer.on("/save",                      HTTP_POST, handleAPSave);
  httpServer.on("/scan",                      HTTP_GET,  handleAPScan);
  // Captive-portal detection endpoints (iOS / Android / Windows)
  httpServer.on("/generate_204",              HTTP_GET,  handleAPRoot);
  httpServer.on("/fwlink",                    HTTP_GET,  handleAPRoot);
  httpServer.on("/connecttest.txt",           HTTP_GET,  handleAPRoot);
  httpServer.on("/hotspot-detect.html",       HTTP_GET,  handleAPRoot);
  httpServer.onNotFound(handleAPRoot);
  httpServer.begin();

  Serial.println("First-boot AP started: " + String(AP_SSID) + " @ " + WiFi.softAPIP().toString());
}

// ── Provisioning page ─────────────────────────────────────────
void handleAPRoot() {
  String html =
    "<!DOCTYPE html><html><head>"
    "<meta name='viewport' content='width=device-width,initial-scale=1'>"
    "<meta charset='UTF-8'>"
    "<title>WEMO First-Time Setup</title>"
    "<style>"
    "*{margin:0;padding:0;box-sizing:border-box}"
    "body{font-family:'Segoe UI',sans-serif;background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);"
         "color:#fff;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:16px}"
    ".card{background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.15);border-radius:18px;"
          "padding:28px 24px;max-width:400px;width:100%;box-shadow:0 8px 32px rgba(0,0,0,.5)}"
    "h1{font-size:1.5em;text-align:center;margin-bottom:4px;"
       "background:linear-gradient(45deg,#00d4ff,#7b2ff7);-webkit-background-clip:text;-webkit-text-fill-color:transparent}"
    ".sub{text-align:center;color:#888;font-size:.82em;margin-bottom:6px}"
    ".badge{text-align:center;margin-bottom:18px;font-size:.75em;padding:6px 12px;"
           "background:rgba(0,255,136,.08);border:1px solid rgba(0,255,136,.2);border-radius:20px;"
           "color:#00ff88;display:inline-block;width:100%}"
    "label{display:block;font-size:.75em;color:#aaa;text-transform:uppercase;letter-spacing:1px;margin:14px 0 5px}"
    "input[type=text],input[type=password]{"
      "width:100%;padding:11px 13px;border-radius:10px;border:1px solid rgba(255,255,255,.15);"
      "background:rgba(255,255,255,.06);color:#fff;font-size:.92em;outline:none}"
    "input:focus{border-color:#00d4ff;background:rgba(0,212,255,.06)}"
    ".btn-connect{width:100%;margin-top:20px;padding:13px;border:none;border-radius:10px;"
                 "background:linear-gradient(45deg,#00d4ff,#7b2ff7);color:#fff;font-size:1em;"
                 "font-weight:700;cursor:pointer;letter-spacing:.5px}"
    ".btn-connect:active{opacity:.8}"
    ".btn-scan{width:100%;margin-top:10px;padding:9px;border:1px solid rgba(255,255,255,.2);"
              "border-radius:8px;background:rgba(255,255,255,.04);color:#aaa;font-size:.83em;cursor:pointer}"
    ".btn-scan:hover{background:rgba(255,255,255,.1);color:#fff}"
    ".net-list{margin-top:8px}"
    ".net{padding:9px 12px;margin:4px 0;background:rgba(255,255,255,.05);border-radius:8px;cursor:pointer;"
         "font-size:.83em;border:1px solid rgba(255,255,255,.07);display:flex;justify-content:space-between;align-items:center}"
    ".net:hover{background:rgba(255,255,255,.1);border-color:#00d4ff}"
    ".rssi{font-size:.73em;color:#888}"
    ".note{margin-top:16px;padding:11px;background:rgba(0,212,255,.07);border-radius:8px;"
          "font-size:.78em;color:#aaa;text-align:center;border:1px solid rgba(0,212,255,.13)}"
    ".ver{text-align:center;margin-top:12px;font-size:.7em;color:#444}"
    "</style></head><body>"
    "<div class='card'>"
      "<h1>&#9889; WEMO Setup</h1>"
      "<div class='sub'>First-time WiFi configuration</div>"
      "<div class='badge'>&#10003; This screen appears only once</div>"
      "<form method='POST' action='/save'>"
        "<label>WiFi Network (SSID)</label>"
        "<input type='text' name='ssid' id='ssidIn' placeholder='Enter or tap a network below' required>"
        "<label>Password</label>"
        "<input type='password' name='pass' id='passIn' placeholder='Leave blank if open network'>"
        "<button type='submit' class='btn-connect'>&#128268; Save &amp; Connect</button>"
      "</form>"
      "<button class='btn-scan' onclick='scan()'>&#128246; Scan Available Networks</button>"
      "<div class='net-list' id='netList'></div>"
      "<div class='note'>&#128274; Credentials saved in device flash.<br>"
           "AP disappears permanently after saving.<br>"
           "Update WiFi later from the WEMO dashboard.</div>"
      "<div class='ver'>WEMO Node v2.4 &nbsp;|&nbsp; 192.168.4.1</div>"
    "</div>"
    "<script>"
    "function scan(){"
      "document.getElementById('netList').innerHTML='<div class=\\'net\\'>Scanning...</div>';"
      "fetch('/scan').then(r=>r.json()).then(nets=>{"
        "let h='';"
        "nets.forEach(n=>{"
          "let bar=n.rssi>-65?'&#9646;&#9646;&#9646;':n.rssi>-80?'&#9646;&#9646;&#9645;':'&#9646;&#9645;&#9645;';"
          "h+='<div class=\"net\" onclick=\"pick(\\'' + n.ssid.replace(/\\'/g,\"\\\\'\")+'\\')\"><span>'+n.ssid+'</span>';"
          "h+='<span class=\"rssi\">'+bar+' '+n.rssi+'dBm</span></div>';"
        "});"
        "document.getElementById('netList').innerHTML=h||'<div class=\\'net\\'>No networks found</div>';"
      "}).catch(()=>document.getElementById('netList').innerHTML='<div class=\\'net\\'>Scan failed</div>');"
    "}"
    "function pick(s){document.getElementById('ssidIn').value=s;document.getElementById('passIn').focus()}"
    "</script>"
    "</body></html>";
  httpServer.send(200, "text/html", html);
}

// ── Save → write NVS → restart (AP never starts again) ────────
void handleAPSave() {
  String ssid = httpServer.arg("ssid");
  String pass = httpServer.arg("pass");
  if (ssid.length() == 0) {
    httpServer.send(400, "text/plain", "SSID required");
    return;
  }
  saveCredentials(ssid, pass);   // write to NVS — AP won't start next boot

  String html =
    "<!DOCTYPE html><html><head>"
    "<meta name='viewport' content='width=device-width,initial-scale=1'>"
    "<title>WEMO Saved</title>"
    "<style>"
    "*{margin:0;padding:0;box-sizing:border-box}"
    "body{font-family:'Segoe UI',sans-serif;background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);"
         "color:#fff;min-height:100vh;display:flex;align-items:center;justify-content:center}"
    ".card{text-align:center;padding:36px 28px;background:rgba(255,255,255,.07);"
          "border-radius:18px;border:1px solid rgba(0,255,136,.3);max-width:360px;width:90%}"
    "h2{color:#00ff88;font-size:1.6em;margin-bottom:12px}"
    "p{color:#aaa;font-size:.88em;line-height:1.6}"
    ".spin{width:38px;height:38px;border:3px solid rgba(255,255,255,.1);border-top-color:#00d4ff;"
          "border-radius:50%;animation:s 1s linear infinite;margin:18px auto}"
    "@keyframes s{to{transform:rotate(360deg)}}"
    "</style></head><body>"
    "<div class='card'>"
      "<h2>&#10003; Saved!</h2>"
      "<div class='spin'></div>"
      "<p>Connecting to <strong style='color:#fff'>" + ssid + "</strong>...</p>"
      "<p style='margin-top:10px;font-size:.8em;color:#555'>"
         "Reconnect your phone to your home WiFi,<br>"
         "then open <strong style='color:#aaa'>http://13.60.236.168</strong></p>"
      "<p style='margin-top:10px;font-size:.75em;color:#444'>"
         "This setup portal will not appear again.</p>"
    "</div></body></html>";
  httpServer.send(200, "text/html", html);
  delay(1500);
  ESP.restart();
}

// ── Network scanner (JSON) ────────────────────────────────────
void handleAPScan() {
  int n = WiFi.scanNetworks(false, true);
  String json = "[";
  for (int k = 0; k < n && k < 20; k++) {
    if (k > 0) json += ",";
    String s = WiFi.SSID(k);
    s.replace("\"", "\\\"");
    json += "{\"ssid\":\"" + s + "\",\"rssi\":" + String(WiFi.RSSI(k)) + "}";
  }
  json += "]";
  httpServer.send(200, "application/json", json);
}

// ══════════════════════════════════════════════════════════════
//  SETUP
// ══════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  bootTime = millis();

  // ── Relay & Button ───────────────────────────────────────────
  // Active-LOW relay: HIGH = OFF, LOW = ON
  // Button works immediately — zero WiFi dependency
  pinMode(RELAY_PIN, OUTPUT);
  pinMode(BTN_PIN,   INPUT_PULLUP);
  digitalWrite(RELAY_PIN, HIGH);   // start safe — relay OFF

  // ── TFT ─────────────────────────────────────────────────────
  tft.initR(INITR_BLACKTAB);
  tft.setRotation(1);
  tft.fillScreen(C_BG);
  tft.setTextSize(1);
  tft.setTextWrap(false);
  drawBootScreen();

  // ── Load credentials from NVS ────────────────────────────────
  loadCredentials();

  // ── Decide: first boot (NVS empty) or normal boot ────────────
  if (wifiSSID.length() == 0) {
    // ── FIRST BOOT — NVS empty — start AP for one-time setup ───
    bootStatus("WIFI", "First boot — AP", C_ORANGE, 30);
    startFirstBootAP();
    bootStatus("WIFI", "WEMO-Setup / .4.1", C_ORANGE, 30);
    // Skip LoRa + MQTT init display — show full boot then drop to loop
    bootDone();
    delay(900);
    return;   // jump straight to loop, which handles AP mode
  }

  // ── NORMAL BOOT — NVS has credentials ────────────────────────
  bootStatus("WIFI", "Connecting...", C_YELLOW, 30);
  wifiOK = connectWiFi(wifiSSID, wifiPass, 15000);

  if (wifiOK) {
    bootStatus("WIFI", WiFi.localIP().toString().c_str(), C_GREEN, 30);
    Serial.println("WiFi connected: " + WiFi.localIP().toString());
  } else {
    bootStatus("WIFI", "Offline mode", C_RED, 30);
    Serial.println("WiFi failed — running offline (no AP started)");
  }

  // ── LoRa ─────────────────────────────────────────────────────
  bootStatus("LORA", "433 MHz init...", C_YELLOW, 44);
  LoRa.setPins(LORA_CS, LORA_RST, LORA_DIO0);
  loraOK = LoRa.begin(433E6);
  bootStatus("LORA", loraOK ? "433 MHz  OK" : "NO MODULE", loraOK ? C_GREEN : C_RED, 44);

  // ── MQTT (only if WiFi up) ───────────────────────────────────
  if (wifiOK) {
    mqtt.setServer(MQTT_HOST, MQTT_PORT);
    mqtt.setCallback(mqttCallback);
    bootStatus("MQTT", "Broker configured", C_CYAN, 58);
  } else {
    bootStatus("MQTT", "Offline mode", C_ORANGE, 58);
  }

  bootDone();
  delay(900);
}

// ══════════════════════════════════════════════════════════════
//  LOOP
// ══════════════════════════════════════════════════════════════
void loop() {

  // ╔══════════════════════════════════════════════════════════╗
  // ║  PHYSICAL BUTTON — ALWAYS FIRST, NO WIFI NEEDED         ║
  // ╚══════════════════════════════════════════════════════════╝
  if (digitalRead(BTN_PIN) == LOW) {
    delay(50);
    if (digitalRead(BTN_PIN) == LOW) {
      relayOn = !relayOn;
      digitalWrite(RELAY_PIN, relayOn ? LOW : HIGH);
      lastBtn = millis();
      Serial.println(relayOn ? "Button → Relay ON" : "Button → Relay OFF");
      if (mqttOK) mqtt.publish("wemo/control", relayOn ? "ON" : "OFF");
      while (digitalRead(BTN_PIN) == LOW);
    }
  }

  // ── FIRST-BOOT AP MODE ───────────────────────────────────────
  // Serves the setup portal. Button still works above.
  // Exits only when user saves credentials → ESP.restart().
  if (apMode) {
    dnsServer.processNextRequest();
    httpServer.handleClient();
    // PZEM + TFT still update so screen isn't blank during setup
    if (millis() - lastTX >= 2000UL) {
      updateSystem();
      lastTX = millis();
    }
    return;   // skip MQTT/LoRa — not needed during first-boot AP
  }

  // ── NORMAL MODE — MQTT ───────────────────────────────────────
  if (wifiOK) {
    if (!mqtt.connected()) {
      mqttOK = false;
      if (mqtt.connect("WEMO_NODE_KU")) {
        mqtt.subscribe("wemo/control");      // relay on/off from dashboard
        mqtt.subscribe("wemo/wifi-config");  // credential update from dashboard
        mqttOK = true;
        Serial.println("MQTT connected");
      }
      // No delay / no return — button check still runs every tick
    } else {
      mqttOK = true;
      mqtt.loop();
    }
  }

  // ── LoRa receive — online AND offline ────────────────────────
  if (loraOK) {
    int pktSize = LoRa.parsePacket();
    if (pktSize) {
      String loraCmd = "";
      while (LoRa.available()) loraCmd += (char)LoRa.read();
      loraCmd.trim();
      Serial.println("LoRa RX: " + loraCmd);
      if (loraCmd == "ON") {
        relayOn = true;
        digitalWrite(RELAY_PIN, LOW);
        lastBtn = millis();
        if (mqttOK) mqtt.publish("wemo/control", "ON");
      } else if (loraCmd == "OFF") {
        relayOn = false;
        digitalWrite(RELAY_PIN, HIGH);
        lastBtn = millis();
        if (mqttOK) mqtt.publish("wemo/control", "OFF");
      }
    }
  }

  // ── Periodic PZEM + TFT + publish ────────────────────────────
  if (millis() - lastTX >= 2000UL) {
    updateSystem();
    lastTX = millis();
  }
}

// ══════════════════════════════════════════════════════════════
//  MQTT CALLBACK
// ══════════════════════════════════════════════════════════════
void mqttCallback(char* topic, byte* payload, unsigned int len) {
  String msg;
  for (unsigned int j = 0; j < len; j++) msg += (char)payload[j];
  msg.trim();

  // ── Relay control from MQTT dashboard ────────────────────────
  if (String(topic) == "wemo/control") {
    if (msg == "ON") {
      relayOn = true;
      digitalWrite(RELAY_PIN, LOW);
      lastBtn = millis();
      Serial.println("Relay ON  via MQTT dashboard");
    } else if (msg == "OFF") {
      relayOn = false;
      digitalWrite(RELAY_PIN, HIGH);
      lastBtn = millis();
      Serial.println("Relay OFF via MQTT dashboard");
    }
    return;
  }

  // ── WiFi credential update from MQTT dashboard ───────────────
  // Dashboard sends: {"ssid":"NewNetwork","pass":"NewPassword"}
  // Device saves to NVS → restarts → connects to new network
  if (String(topic) == "wemo/wifi-config") {
    StaticJsonDocument<200> cfg;
    if (!deserializeJson(cfg, msg) && cfg.containsKey("ssid")) {
      String newSSID = cfg["ssid"].as<String>();
      String newPass = cfg["pass"] | "";
      if (newSSID.length() > 0) {
        Serial.println("WiFi update via dashboard → " + newSSID);
        saveCredentials(newSSID, newPass);
        delay(500);
        ESP.restart();
      }
    }
  }
}

// ══════════════════════════════════════════════════════════════
//  BOOT SCREEN HELPERS
// ══════════════════════════════════════════════════════════════
void drawBootScreen() {
  tft.fillRect(0, 0, 160, 14, C_BLUE);
  tft.setCursor(4, 4); tft.setTextColor(C_CYAN);
  tft.print("WEMO  NODE  v2.4");
  tft.drawRect(1, 16, 158, 84, C_BORDER);
  tft.setCursor(4, 20); tft.setTextColor(C_GRAY);
  tft.print("Initialising subsystems...");
}

void bootStatus(const char* sys, const char* msg, uint16_t col, int y) {
  tft.fillRect(2, y, 156, 10, C_PANEL);
  tft.setCursor(5, y+1);  tft.setTextColor(C_GRAY); tft.print(sys);
  tft.setCursor(30, y+1); tft.setTextColor(col);    tft.print(msg);
}

void bootDone() {
  tft.fillRect(2, 72, 156, 10, C_PANEL);
  tft.setCursor(20, 73); tft.setTextColor(C_GREEN);
  tft.print("System Ready  >>>");
}

// ══════════════════════════════════════════════════════════════
//  UPTIME FORMATTER
// ══════════════════════════════════════════════════════════════
String fmtUptime(unsigned long ms) {
  unsigned long s = ms / 1000;
  if (s < 3600) return String(s / 60) + "m" + String(s % 60) + "s";
  return String(s / 3600) + "h" + String((s % 3600) / 60) + "m";
}

// ══════════════════════════════════════════════════════════════
//  UPDATE SYSTEM — PZEM read, TFT draw, MQTT + LoRa + Flask publish
// ══════════════════════════════════════════════════════════════
void updateSystem() {
  float v  = pzem.voltage();
  float i  = pzem.current();
  float p  = pzem.power();
  float e  = pzem.energy();
  float f  = pzem.frequency();
  float pf = pzem.pf();

  pzemOK = !isnan(v);

  if (isnan(v))  v  = 0.0f;
  if (isnan(i))  i  = 0.0f;
  if (isnan(p))  p  = 0.0f;
  if (isnan(e))  e  = 0.0f;
  if (isnan(f))  f  = 0.0f;
  if (isnan(pf)) pf = 0.0f;

  if (!relayOn) { i = 0.0f; p = 0.0f; pf = 0.0f; f = 0.0f; }
  if (v  < 20.0f)                          v  = 0.0f;
  if (i  < 0.05f)                          i  = 0.0f;
  if (p  < 3.0f)                           p  = 0.0f;
  if (relayOn && (f < 40.0f || f > 65.0f)) f  = 0.0f;
  if (i == 0.0f) { p = 0.0f; pf = 0.0f; }

  String up = fmtUptime(millis() - bootTime);
  drawTFT(v, i, p, e, f, pf, up);

  // ── MQTT publish — only when online ──────────────────────────
  if (wifiOK && mqttOK) {
    StaticJsonDocument<256> doc;
    doc["v"]      = v;   doc["i"]  = i;   doc["p"]  = p;
    doc["e"]      = e;   doc["f"]  = f;   doc["pf"] = pf;
    doc["relay"]  = relayOn ? 1 : 0;
    doc["rssi"]   = WiFi.RSSI();
    doc["uptime"] = up;
    char buf[256];
    serializeJson(doc, buf);
    if (mqtt.publish("wemo/data", buf))
      Serial.println("TX: " + String(buf));
  }

  // ── Flask HTTP POST — power + pf only ────────────────────────
  // Sends {"power":<W>, "pf":<0-1>} to FLASK_URL every cycle.
  // Non-blocking: 2-second timeout; failure is silently logged.
  if (wifiOK) {
    HTTPClient http;
    http.begin(FLASK_URL);
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(2000);   // 2 s — won't block PZEM/TFT cycle

    StaticJsonDocument<320> fdoc;
    fdoc["v"]      = v;
    fdoc["i"]      = i;
    fdoc["power"]  = p;
    fdoc["pf"]     = pf;
    fdoc["e"]      = e;
    fdoc["f"]      = f;
    fdoc["relay"]  = relayOn ? 1 : 0;
    fdoc["rssi"]   = WiFi.RSSI();
    fdoc["uptime"] = up;

    char fbuf[256];   // must be large enough for all fields (~170 bytes)
    serializeJson(fdoc, fbuf);

    int httpCode = http.POST(fbuf);
    if (httpCode > 0) {
      Serial.println("Flask POST " + String(httpCode) + " → " + String(fbuf));
    } else {
      Serial.println("Flask POST failed: " + http.errorToString(httpCode));
    }
    http.end();
  }


  // ── Flask relay-state poll — dashboard relay control ──────────
  if (wifiOK) {
    HTTPClient httpRelay;
    httpRelay.begin(FLASK_RELAY_URL);
    httpRelay.setTimeout(2000);
    int rCode = httpRelay.GET();
    if (rCode == 200) {
      String body = httpRelay.getString();
      StaticJsonDocument<64> rdoc;
      if (!deserializeJson(rdoc, body)) {
        int desired = rdoc["relay"] | -1;
        if (desired == 1 && !relayOn) {
          relayOn = true;
          digitalWrite(RELAY_PIN, LOW);
          Serial.println("[RELAY] Dashboard → ON");
        } else if (desired == 0 && relayOn) {
          relayOn = false;
          digitalWrite(RELAY_PIN, HIGH);
          Serial.println("[RELAY] Dashboard → OFF");
        }
      }
    }
    httpRelay.end();
  }

  // ── LoRa TX — online AND offline ─────────────────────────────
  if (loraOK) {
    StaticJsonDocument<256> doc;
    doc["v"] = v; doc["i"] = i; doc["p"] = p;
    doc["e"] = e; doc["f"] = f; doc["pf"] = pf;
    doc["relay"] = relayOn ? 1 : 0;
    char buf[256];
    serializeJson(doc, buf);
    LoRa.beginPacket();
    LoRa.print(buf);
    LoRa.endPacket();
  }
}

// ══════════════════════════════════════════════════════════════
//  TFT DRAWING HELPERS
// ══════════════════════════════════════════════════════════════
void drawBox(int x, int y, int w, int h, uint16_t col) {
  tft.drawRect(x, y, w, h, col);
}

void drawBar(int x, int y, int w, int h, float ratio, uint16_t col) {
  tft.fillRect(x, y, w, h, C_MGRAY);
  int fw = (int)(constrain(ratio, 0.0f, 1.0f) * w);
  if (fw > 0) tft.fillRect(x, y, fw, h, col);
}

void drawDot(int x, int y, bool ok, uint16_t okCol) {
  tft.fillRect(x-2, y-2, 5, 5, C_BG);
  tft.fillCircle(x, y, 2, ok ? okCol : C_RED);
}

// ══════════════════════════════════════════════════════════════
//  TFT FULL LAYOUT  — v2.4
//
//  Status row:  MQ/LR/PZ/BT dots  |  WiFi CONN+RSSI / NO WiFi
//  Footer:      Global: ON x.x.x  |  Global: OFF | Btn+LoRa OK
//  During first-boot AP: shows "SETUP" banner in header
// ══════════════════════════════════════════════════════════════
void drawTFT(float v, float i, float p, float e, float f, float pf, const String& up) {
  tft.fillScreen(C_BG);
  tft.setTextSize(1);
  tft.setTextWrap(false);

  // ── Header bar ───────────────────────────────────────────────
  tft.fillRect(0, 0, 160, 14, 0x000F);
  tft.setCursor(3, 3); tft.setTextColor(C_CYAN);
  tft.print("WEMO");

  const char* dlbls[3] = {"M", "L", "P"};
  bool        doks[3]  = {mqttOK, loraOK, pzemOK};
  for (int d = 0; d < 3; d++) {
    int bx = 34 + d * 16;
    tft.setCursor(bx, 3); tft.setTextColor(C_MGRAY); tft.print(dlbls[d]);
    drawDot(bx + 9, 7, doks[d]);
  }

  if (apMode) {
    tft.setCursor(88, 3); tft.setTextColor(C_ORANGE); tft.print("SETUP AP");
  } else {
    tft.setCursor(88, 3); tft.setTextColor(C_GRAY); tft.print(up);
  }

  // ── Voltage card ─────────────────────────────────────────────
  drawBox(1, 15, 50, 49, C_BORDER);
  tft.fillRect(2, 16, 48, 47, C_PANEL);
  tft.fillRect(2, 16, 48, 8, 0x2000);
  tft.setCursor(4, 18);  tft.setTextColor(C_YELLOW); tft.print(" VOLTAGE");
  tft.fillRect(4, 28, 44, 18, C_PANEL);
  tft.setCursor(4, 28);  tft.setTextColor(C_YELLOW); tft.setTextSize(2);
  tft.print(String((int)v));           // ← FIX: integer only — "233" fits, "233.1" clipped
  tft.setTextSize(1);
  tft.setCursor(4, 48);  tft.setTextColor(C_MGRAY); tft.print("Volts AC");
  drawBar(4, 57, 44, 3, v / MAX_VOLTS, C_YELLOW);

  // ── Current card ─────────────────────────────────────────────
  drawBox(53, 15, 50, 49, C_BORDER);
  tft.fillRect(54, 16, 48, 47, C_PANEL);
  tft.fillRect(54, 16, 48, 8, 0x1008);
  tft.setCursor(56, 18); tft.setTextColor(C_PURPLE); tft.print(" CURRENT");
  tft.fillRect(56, 28, 44, 18, C_PANEL);
  tft.setCursor(56, 28); tft.setTextColor(C_CYAN); tft.setTextSize(2);
  tft.print(i < 10 ? String(i, 2) : String(i, 1));
  tft.setTextSize(1);
  tft.setCursor(56, 48); tft.setTextColor(C_MGRAY); tft.print("Amps RMS");
  drawBar(56, 57, 44, 3, i / MAX_AMPS, C_CYAN);

  // ── Power card ───────────────────────────────────────────────
  drawBox(105, 15, 54, 49, C_BORDER);
  tft.fillRect(106, 16, 52, 47, C_PANEL);
  tft.fillRect(106, 16, 52, 8, 0x0010);
  tft.setCursor(108, 18); tft.setTextColor(C_SKYBLUE); tft.print("  POWER");
  tft.fillRect(108, 28, 48, 18, C_PANEL);
  tft.setCursor(108, 28); tft.setTextColor(C_GREEN); tft.setTextSize(2);
  tft.print(p < 1000 ? String(p, 1) : String(p, 0));
  tft.setTextSize(1);
  tft.setCursor(108, 48); tft.setTextColor(C_MGRAY); tft.print("Watts");
  drawBar(108, 57, 48, 3, p / MAX_WATTS, C_GREEN);

  // ── Hz / PF / Load row ───────────────────────────────────────
  drawBox(1, 65, 158, 12, C_BORDER);
  tft.fillRect(2, 66, 156, 10, C_PANEL);
  tft.setCursor(4, 68);  tft.setTextColor(C_MGRAY); tft.print("Hz");
  tft.fillRect(17, 66, 40, 10, C_PANEL);
  tft.setCursor(17, 68); tft.setTextColor(C_ORANGE); tft.print(String(f, 1));
  tft.drawFastVLine(60, 66, 10, C_BORDER);
  tft.setCursor(63, 68); tft.setTextColor(C_MGRAY); tft.print("PF");
  tft.fillRect(77, 66, 30, 10, C_PANEL);
  tft.setCursor(77, 68); tft.setTextColor(C_ORANGE); tft.print(String(pf, 2));
  tft.drawFastVLine(110, 66, 10, C_BORDER);
  float ld = constrain((p / MAX_WATTS) * 100.0f, 0.0f, 100.0f);
  tft.setCursor(113, 68); tft.setTextColor(C_MGRAY); tft.print("LD");
  tft.fillRect(127, 66, 29, 10, C_PANEL);
  tft.setCursor(127, 68);
  tft.setTextColor(ld >= 80 ? C_RED : ld >= 50 ? C_YELLOW : C_GREEN);
  tft.print(String(ld, 0) + "%");

  // ── Energy row ───────────────────────────────────────────────
  drawBox(1, 78, 158, 12, C_BORDER);
  tft.fillRect(2, 79, 156, 10, C_PANEL);
  tft.setCursor(4, 81);  tft.setTextColor(C_MGRAY);  tft.print("ENERGY");
  tft.fillRect(46, 79, 60, 10, C_PANEL);
  tft.setCursor(46, 81); tft.setTextColor(C_ORANGE);
  if (e < 1.0f) tft.print(String(e * 1000.0f, 2) + " Wh");
  else          tft.print(String(e, 4) + " kWh");
  drawBar(112, 82, 44, 3, constrain(e / 10.0f, 0.0f, 1.0f), C_ORANGE);

  // ── Relay status banner ──────────────────────────────────────
  uint16_t relBg = relayOn ? C_DKGRN : C_DKRED;
  uint16_t relFg = relayOn ? C_GREEN  : C_RED;
  tft.fillRect(1, 91, 158, 11, relBg);
  tft.drawRect(1, 91, 158, 11, relayOn ? C_GREEN : C_RED);
  String relStr = relayOn ? ">>> RELAY: ACTIVE <<<" : ">>> RELAY: OFFLINE <<<";
  int rx = (160 - (int)relStr.length() * 6) / 2;
  tft.setCursor(rx, 94); tft.setTextColor(relFg);
  tft.print(relStr);

  // ── Status row ───────────────────────────────────────────────
  //  Left:  MQ | LR | PZ | BT  dots
  //  Right: WiFi CONN + RSSI  /  NO WiFi  /  SETUP AP
  drawBox(1, 103, 158, 12, C_BORDER);
  tft.fillRect(2, 104, 156, 10, C_PANEL);

  bool btnRecent = (millis() - lastBtn < 5000UL);
  const char* slbls[4] = {"MQ", "LR", "PZ", "BT"};
  bool        soks[4]  = {mqttOK, loraOK, pzemOK, btnRecent};
  for (int d = 0; d < 4; d++) {
    tft.setCursor(4 + d * 18, 105); tft.setTextColor(C_MGRAY);
    tft.print(slbls[d]);
    drawDot(15 + d * 18, 109, soks[d]);
  }
  tft.drawFastVLine(78, 104, 10, C_BORDER);

  if (apMode) {
    tft.setCursor(81, 105); tft.setTextColor(C_ORANGE); tft.print("AP .4.1");
  } else if (wifiOK) {
    int rssi = WiFi.RSSI();
    uint16_t rssiCol = rssi > -65 ? C_GREEN : rssi > -80 ? C_YELLOW : C_RED;
    tft.setCursor(81, 105); tft.setTextColor(C_GREEN);   tft.print("WiFi");
    tft.setCursor(105, 105); tft.setTextColor(rssiCol);  tft.print(String(rssi));
  } else {
    tft.setCursor(81, 105); tft.setTextColor(C_RED); tft.print("NO WiFi");
  }

  // ── Footer — Global Access indicator ─────────────────────────
  tft.fillRect(0, 116, 160, 12, 0x0208);
  tft.setCursor(4, 119); tft.setTextColor(C_MGRAY); tft.print("Global:");

  if (apMode) {
    tft.setCursor(46, 119); tft.setTextColor(C_ORANGE); tft.print("SETUP — connect phone");
  } else if (wifiOK) {
    tft.setCursor(46, 119); tft.setTextColor(C_GREEN); tft.print("ON");
    String ip   = WiFi.localIP().toString();
    int    dot1 = ip.lastIndexOf('.', ip.lastIndexOf('.') - 1);
    tft.setCursor(62, 119); tft.setTextColor(C_CYAN); tft.print(ip.substring(dot1));
  } else {
    tft.setCursor(46, 119); tft.setTextColor(C_RED);   tft.print("OFF");
    tft.setCursor(70, 119); tft.setTextColor(C_MGRAY); tft.print("| Btn+LoRa OK");
  }
}
