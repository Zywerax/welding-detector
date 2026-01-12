# 🎥 Welding Detector - Szczegółowa Dokumentacja

## 📖 Spis treści
1. [Co to jest i do czego służy?](#co-to-jest)
2. [Jak to działa - architektura](#jak-to-działa)
3. [Główne funkcje aplikacji](#główne-funkcje)
4. [Użyte technologie i biblioteki](#technologie)
5. [Problemy z którymi się mierzyliśmy](#problemy)
6. [Struktura projektu](#struktura)
7. [Jak uruchomić aplikację](#uruchomienie)

---

## 🎯 Co to jest i do czego służy? {#co-to-jest}

**Welding Detector** to inteligentna aplikacja do monitorowania i automatycznej kontroli jakości procesu spawania laserowego. Wyobraź sobie system, który:

- 📹 **Nagrywa proces spawania** z kamery USB w czasie rzeczywistym
- 🤖 **Automatycznie wykrywa wady** spawów używając sztucznej inteligencji
- ✂️ **Wycina niepotrzebne fragmenty** filmów (np. sam moment spawania)
- 📊 **Analizuje całe nagrania** i pokazuje statystyki: ile spawów było OK, ile NOK
- 🏷️ **Pozwala oznaczać klatki** i trenować własne modele AI

Jest to system wizyjny do **kontroli jakości produkcji** - zamiast człowieka przeglądającego każdy spaw, AI robi to automatycznie i błyskawicznie.

---

## 🏗️ Jak to działa - Architektura {#jak-to-działa}

Aplikacja składa się z **dwóch głównych części**:

### 1️⃣ Backend (Serwer) - Python
To "silnik" aplikacji, który robi całą ciężką pracę:
- Łączy się z kamerą USB i przechwytuje obraz
- Nagrywa wideo do plików MP4
- Uruchamia modele AI do wykrywania wad
- Przetwarza filmy (wykrywa ruch, wycina spawanie)
- Odpowiada na żądania frontendu przez API

**Technologia:** Python + FastAPI (szybki framework webowy)

### 2️⃣ Frontend (Interfejs użytkownika) - Vue.js
To strona internetowa, którą widzisz w przeglądarce:
- Pokazuje obraz z kamery na żywo (live stream)
- Ma przyciski do nagrywania, analizy, przeglądania wyników
- Wyświetla ładne wykresy i statystyki
- Pozwala oznaczać zdjęcia spawów (OK/NOK)

**Technologia:** Vue.js 3 + Vite + TailwindCSS (nowoczesny stack frontendowy)

### 🔄 Jak się komunikują?

```
┌─────────────────────┐          HTTP/REST API           ┌──────────────────────┐
│   Frontend (Vue)    │ ←─────────────────────────────→ │   Backend (Python)   │
│  (przeglądarka)     │                                  │      FastAPI         │
│                     │  GET /camera/stream - pobierz    │                      │
│  - Live stream      │  POST /recording/start - nagraj  │  - Kamera USB        │
│  - Przyciski        │  POST /ml/analyze - analizuj     │  - Nagrywanie MP4    │
│  - Wyniki analizy   │  GET /recording/list - lista     │  - Modele AI         │
└─────────────────────┘                                  └──────────────────────┘
```

Frontend wysyła **żądania HTTP** (jak: "daj mi listę nagrań"), a backend odpowiada **JSON** (strukturą danych).

---

## 🚀 Główne funkcje aplikacji {#główne-funkcje}

### 📹 1. Live Streaming i nagrywanie

**Co robi:**
- Pokazuje obraz z kamery USB w czasie rzeczywistym
- Pozwala nagrywać wideo do pliku MP4
- Nakłada timestamp (datę i godzinę) oraz znacznik "REC" na obraz

**Jak to działa technicznie:**

1. **Background Capture Thread** - osobny wątek w Pythonie, który non-stop pobiera klatki z kamery:
```python
while running:
    ret, frame = cap.read()  # Pobierz klatkę z kamery
    # Konwertuj do JPEG
    _, buf = cv2.imencode('.jpg', frame, quality=95)
    last_frame = buf.tobytes()  # Zapisz do bufora
```

2. **MJPEG Streaming** - każda klatka jest wysyłana jako JPEG przez HTTP:
```
--frame
Content-Type: image/jpeg

[DANE JPEG KLATKI]
--frame
Content-Type: image/jpeg

[NASTĘPNA KLATKA]
...
```

3. **Nagrywanie do MP4** - podczas nagrywania klatki są zapisywane przez OpenCV:
```python
video_writer = cv2.VideoWriter('nagranie.mp4', codec='mp4v', fps=30)
video_writer.write(frame)  # Zapisz każdą klatkę
```

**Problem który rozwiązaliśmy:**
- Pierwsze wersje miały **opóźnienie** (lag) 3-5 sekund
- **Rozwiązanie:** Użycie backend MSMF (Media Foundation) + format MJPEG + bufor 1 klatka
- Wynik: Opóźnienie spadło do ~0.1s

### 🤖 2. Automatyczna analiza wad (Machine Learning)

**Co robi:**
- Analizuje każdą klatkę wideo i określa: **OK** (dobry spaw) lub **NOK** (wada)
- Jeśli NOK, klasyfikuje **typ wady**: pęknięcie, porowatość, brak przetopu, itp.
- Pokazuje **procentową pewność** przewidywania (np. NOK 95%)

**Jak to działa:**

1. **Model binarny (OK/NOK)** - EfficientNet-B0:
```
Zdjęcie spawu → Sieć neuronowa → [0.85 OK, 0.15 NOK] → Wynik: OK (85%)
```

2. **Model defektów (typ wady)** - też EfficientNet-B0:
```
Zdjęcie wady → Sieć neuronowa → [crack: 0.92, porosity: 0.05, ...] → Wynik: pęknięcie (92%)
```

**Co to jest EfficientNet-B0?**
To gotowa architektura sieci neuronowej (z biblioteki `timm`), która jest:
- **Szybka** - może analizować wiele zdjęć na sekundę
- **Dokładna** - nauczona na milionach zdjęć (ImageNet)
- **Mała** - waży ~20MB, działa nawet na CPU

**Co to jest Grad-CAM?**
To wizualizacja "na co patrzy AI" - nakłada heatmapę pokazującą, które fragmenty zdjęcia wpłynęły na decyzję:

```
Oryginalne zdjęcie + Heatmapa = Widać gdzie AI znalazł wadę (np. pęknięcie świeci na czerwono)
```

**Trenowanie modelu:**
- Użytkownik oznacza klatki (🏷️ Labeling) jako OK/NOK + typ wady
- Po zebraniu np. 100+ zdjęć klikasz "🏋️ Trenuj model"
- Aplikacja uruchamia trening (10-30 minut na CPU, 2-5 min na GPU)
- Nowy model jest zapisywany i od razu używany do przewidywań

### 📊 3. Batch analiza nagrań

**Co robi:**
- Analizuje **całe wideo** klatka po klatce
- Tworzy raport: ile OK, ile NOK, jakie typy wad
- Pokazuje miniatury wszystkich klatek NOK
- Wyniki są zapisywane i widoczne nawet po odświeżeniu strony

**Jak to działa:**

1. Klikasz "🔬 Analizuj wideo"
2. Backend:
```python
for frame in video:
    prediction = ml_model.predict(frame)  # OK czy NOK?
    if prediction == "nok":
        defect_type = defect_model.predict(frame)  # Jaki typ wady?
    results.append(...)
```
3. Wyniki są zapisywane do `recordings/analysis/{filename}.json`
4. Frontend odczytuje i wyświetla statystyki + miniatury

**Optymalizacja:**
- Parametr `skip_frames=5` - analizuj co 5. klatkę (30x szybciej, prawie taka sama dokładność)

**Przechowywanie wyników:**
Aby wyniki nie znikały po odświeżeniu strony, używamy **localStorage** w przeglądarce:
```javascript
localStorage.setItem('analysisResults', JSON.stringify(wyniki))
// Po odświeżeniu:
wyniki = JSON.parse(localStorage.getItem('analysisResults'))
```

### ✂️ 4. Trim to Motion - Wycinanie zbędnych fragmentów

**Co robi:**
- Automatycznie wykrywa momenty **ruchu** w nagraniu
- Wycina statyczne fragmenty (kiedy nic się nie dzieje)
- Zapisuje krótsze wideo z samą akcją

**Jak wykrywa ruch:**

1. Porównuje sąsiednie klatki:
```python
previous_frame = klatka[0]
current_frame = klatka[1]
difference = abs(current_frame - previous_frame)  # Różnica pikseli
if difference > threshold:
    # Ruch wykryty!
```

2. Grupuje klatki z ruchem w **segmenty**:
```
Klatki: [0..10 statyczne] [11..50 RUCH] [51..60 statyczne] [61..100 RUCH]
Segmenty: [(11, 50), (61, 100)]
```

3. Zapisuje tylko segmenty z ruchem do nowego pliku MP4

**Padding:** Dodaje 30 klatek przed ruchem i 5 po (żeby nie uciąć za wcześnie/późno)

### 🔥 5. Trim to Post-Processing - Usuwanie spawania

**Co robi:**
- Wykrywa moment **aktywnego spawania** (jasny laser)
- **Wycina TYLKO spawanie**, zostawia przygotowanie i inspekcję
- Idealny do przeglądania gotowych spawów bez oślepiającego lasera

**Jak wykrywa spawanie:**

Analizuje każdą klatkę pod kątem jasności i koloru:

```python
# Metoda 1: Bardzo jasne piksele (białe/żółte centrum lasera)
very_bright_pixels = count(pixels > 220)

# Metoda 2: Czerwone/pomarańczowe światło (rozżarzony metal)
red_hot_pixels = count(R>220 AND G>180 AND B<120)

# Jeśli którykolwiek warunek spełniony = spawanie
if very_bright_pixels >= 1% OR red_hot_pixels >= 3%:
    welding_detected = True
```

**Grupowanie z tolerancją:**
```
Gap tolerance = 10 klatek (0.3s)

Jasne klatki: [10, 11, 12, 25, 26, 27] ← przerwa 13 klatek
Segmenty: [(10, 12), (25, 27)]  ← dwa osobne segmenty

Jasne klatki: [10, 11, 12, 18, 19, 20] ← przerwa 6 klatek
Segmenty: [(10, 20)]  ← jeden ciągły segment
```

**Przypadki brzegowe:**
- Nie wykryto spawania → zachowaj całe wideo
- >80% wideo to spawanie → zachowaj drugą połowę (inspekcja)
- Spawanie na końcu → zachowaj początek (przygotowanie)

**Problem który rozwiązaliśmy:**
- Początkowo algorytm wykrywał **czerwoną poświatę po zgaśnięciu lasera** jako spawanie
- Zbyt dużo materiału było wycinane zaraz po spawaniu
- **Rozwiązania:**
  - Zmniejszono gap_tolerance z 30 do 10 klatek (wykrywanie kończy się szybciej)
  - Usunięto buffer na końcu spawania (dokładny moment zgaśnięcia)
  - Zwiększono progi jasności (tylko bardzo jasne światło = aktywny laser)

### 🏷️ 6. Labeling - Oznaczanie danych treningowych

**Co robi:**
- Pozwala ręcznie oznaczać klatki jako OK/NOK
- Dla NOK można wybrać typ wady z 9 kategorii
- Zbiera dane do treningu modeli AI

**Typy wad:**
1. 🫧 Porowatość (porosity) - pęcherzyki powietrza
2. 💔 Pęknięcie (crack) - rysy, szczeliny
3. 🔗 Brak przetopu (lack_of_fusion) - materiał się nie połączył
4. 📉 Podtopienie (undercut) - wgłębienie
5. 🔥 Przepalenie (burn_through) - dziura
6. 💦 Rozpryski (spatter) - rozbryzgi metalu
7. 〰️ Nierówna spoina (irregular_bead)
8. 🦠 Zanieczyszczenie (contamination)
9. ❓ Inna wada (other)

**Workflow:**
```
1. Otwórz Frame Viewer
2. Wybierz klatkę
3. Kliknij "OK" lub "NOK"
4. Jeśli NOK → wybierz typ wady
5. Auto-przejście do następnej klatki
6. Po 100+ oznaczonych → trenuj model
```

### ⚙️ 7. Ustawienia kamery

**Co można zmienić:**
- **Rozdzielczość:** 1280x720 (HD) lub 1920x1080 (FHD)
- **FPS:** 30 lub 60 klatek na sekundę
- **Jakość JPEG:** 50-100% (wpływa na rozmiar pliku)
- **Kontrast:** 0-255 (dla ciemnych/jasnych scen)
- **Monochromatyczny:** czarno-biały obraz

**Rzeczywisty vs. Żądany FPS:**

Aplikacja **mierzy rzeczywisty FPS** kamery:
```python
start = time.now()
for i in range(60):
    cap.read()  # Pobierz 60 klatek
elapsed = time.now() - start
actual_fps = 60 / elapsed  # Np. 60 / 2.1s = 28.5 FPS
```

Dlaczego? Kamera może nie wspierać żądanych 60 FPS - wtedy dostajemy np. 30 FPS. Musimy to wiedzieć, żeby poprawnie zapisać wideo (inaczej odtwarzane jest za szybko/wolno).

---

## 🛠️ Technologie i biblioteki {#technologie}

### Backend (Python)

#### 1. **FastAPI** - Framework webowy
```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/camera/stream")
async def stream():
    return StreamingResponse(...)
```
**Po co:** Tworzenie REST API - endpointów, które odpowiadają na żądania HTTP.
- Szybki (async/await)
- Automatyczna dokumentacja (Swagger UI)
- Walidacja typów (Pydantic)

#### 2. **OpenCV (cv2)** - Przetwarzanie obrazu i wideo
```python
import cv2
cap = cv2.VideoCapture(0)  # Otwórz kamerę
ret, frame = cap.read()     # Pobierz klatkę
cv2.imwrite('frame.jpg', frame)  # Zapisz jako JPEG
```
**Po co:** Wszystko związane z kamerą i wideo:
- Przechwytywanie z USB
- Kodowanie/dekodowanie JPEG, MP4
- Przetwarzanie obrazu (konwersja kolorów, blur, threshold)
- Wykrywanie ruchu (absdiff)

#### 3. **PyTorch** - Deep Learning
```python
import torch
model = torch.load('model.pth')
prediction = model(image)  # [0.85 OK, 0.15 NOK]
```
**Po co:** Uruchamianie sieci neuronowych do klasyfikacji.
- GPU acceleration (CUDA)
- Automatyczne różniczkowanie (autograd)
- Transfer learning (wykorzystanie gotowych modeli)

#### 4. **timm** (PyTorch Image Models)
```python
import timm
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
```
**Po co:** Biblioteka z gotowymi architekturami sieci (EfficientNet, ResNet, Vision Transformer).
- Pretrained weights (wytrenowane na ImageNet)
- Setki modeli "out of the box"

#### 5. **torchvision** - Transformacje obrazów
```python
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406])
])
```
**Po co:** Przygotowanie obrazów do sieci neuronowej (resize, normalizacja).

#### 6. **NumPy** - Obliczenia numeryczne
```python
import numpy as np
arr = np.array([1, 2, 3])
mean = np.mean(arr)  # 2.0
```
**Po co:** Szybkie operacje na tablicach (klatki wideo to tablice NumPy).

#### 7. **Uvicorn** - Serwer ASGI
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
**Po co:** Uruchamia aplikację FastAPI.

---

### Frontend (Vue.js)

#### 1. **Vue.js 3** - Framework JavaScript
```vue
<template>
  <button @click="startRecording">🔴 Nagrywaj</button>
</template>

<script setup>
function startRecording() {
  fetch('/recording/start', { method: 'POST' })
}
</script>
```
**Po co:** Tworzenie reaktywnego UI.
- Reactive state (ref, reactive)
- Komponenty (modularne UI)
- Two-way binding (v-model)

#### 2. **Vite** - Build tool
```bash
npm run dev  # Szybki dev server z HMR
npm run build  # Produkcyjny build
```
**Po co:** Szybkie budowanie i hot reload podczas developmentu.

#### 3. **TailwindCSS** - Style CSS
```html
<button class="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded">
  Kliknij mnie
</button>
```
**Po co:** Utility-first CSS - szybkie stylowanie bez pisania CSS.

#### 4. **Fetch API** - HTTP requesty
```javascript
const response = await fetch('/api/endpoint')
const data = await response.json()
```
**Po co:** Komunikacja z backendem (pobieranie/wysyłanie danych).

---

## 🐛 Problemy z którymi się mierzyliśmy {#problemy}

### Problem 1: Opóźnienie stream kamery (3-5 sekund lag)

**Objawy:**
- Ruszasz ręką przed kamerą, a na ekranie pojawia się to 3 sekundy później
- Niemożliwość precyzyjnego pozycjonowania

**Przyczyna:**
- Domyślny backend OpenCV (Auto) buforował wiele klatek
- Kamera wysyłała raw BGR, co było wolne przez USB

**Rozwiązanie:**
1. **Zmiana backend na MSMF** (Media Foundation):
```python
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)  # Najszybszy na Windows
```

2. **Format MJPEG** (sprzętowa kompresja):
```python
cap.set(cv2.CAP_PROP_FOURCC, 'MJPG')  # Kamera kompresuje JPEG, nie CPU
```

3. **Minimalny bufor**:
```python
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Tylko 1 klatka w buforze
```

**Wynik:** Lag spadł z 3-5s do ~0.1s ✅

---

### Problem 2: Nieprawidłowa prędkość odtwarzania nagrań

**Objawy:**
- Nagranie odtwarza się za szybko (jak w przyspieszeniu)
- Żądane 30 FPS, a film leci jak 60 FPS

**Przyczyna:**
- Kamera deklaruje 60 FPS, ale faktycznie daje 30 FPS
- VideoWriter zapisuje z żądanym FPS, nie rzeczywistym

**Rozwiązanie:**
Pomiar rzeczywistego FPS:
```python
def _measure_actual_fps():
    start = time.perf_counter()
    frames = 0
    for _ in range(60):
        if cap.read()[0]:
            frames += 1
    elapsed = time.perf_counter() - start
    actual_fps = frames / elapsed
```

Użycie zmierzonego FPS przy nagrywaniu:
```python
writer = cv2.VideoWriter('video.mp4', codec, self.actual_fps, (width, height))
```

**Wynik:** Nagrania odtwarzają się z prawidłową prędkością ✅

---

### Problem 3: Procentowe pewności pokazywały 1000%

**Objawy:**
- Model zwraca 85% pewności
- Na ekranie: "850%" lub "1000%"

**Przyczyna:**
- Model PyTorch zwraca `confidence` w zakresie 0-100
- Frontend dodatkowo mnożył * 100

**Rozwiązanie:**
```javascript
// ❌ Było:
{{ prediction.confidence * 100 }}%

// ✅ Jest:
{{ prediction.confidence }}%
```

**Wynik:** Poprawne wyświetlanie procentów ✅

---

### Problem 4: Wyniki analizy znikały po odświeżeniu strony

**Objawy:**
- Przeanalizujesz wideo, wszystko działa
- Odświeżysz stronę F5 → wyniki zniknęły

**Przyczyna:**
- Wyniki były trzymane tylko w zmiennej `recordings.value` (pamięć RAM)
- Po odświeżeniu strony wszystko się resetowało

**Rozwiązanie:**
Zapisywanie do **localStorage** przeglądarki:
```javascript
// Po zakończeniu analizy:
function saveAnalysisResults() {
  const data = {}
  recordings.value.forEach(rec => {
    if (rec.analysis?.results) {
      data[rec.filename] = rec.analysis
    }
  })
  localStorage.setItem('analysisResults', JSON.stringify(data))
}

// Po załadowaniu strony:
function restoreAnalysisResults() {
  const saved = localStorage.getItem('analysisResults')
  if (saved) {
    const data = JSON.parse(saved)
    recordings.value.forEach(rec => {
      if (data[rec.filename]) {
        rec.analysis = data[rec.filename]
      }
    })
  }
}
```

**Wynik:** Wyniki przetrwają odświeżenie i zamknięcie przeglądarki ✅

---

### Problem 5: Wykrywanie spawania wycinało za dużo (czerwona poświata)

**Objawy:**
- Po zgaśnięciu lasera metal jeszcze czerwono świeci
- Algorytm wykrywał to jako spawanie i wycinał
- Za mało materiału po spawaniu do inspekcji

**Przyczyna:**
- Gap tolerance = 30 klatek (1s) → algorytm kontynuował wykrywanie przez 1s po ostatniej jasnej klatce
- Buffer +3 klatki na końcu spawania
- Niskie progi detekcji (każda czerwona poświata = spawanie)

**Rozwiązanie (iteracyjny proces):**

**Iteracja 1:** Zmniejsz gap_tolerance
```python
gap_tolerance = 30  # Było
gap_tolerance = 10  # Jest (0.3s zamiast 1s)
```

**Iteracja 2:** Usuń buffer na końcu
```python
# Było:
weld_end_buffered = weld_end + buffer_frames

# Jest:
weld_end_buffered = weld_end  # Dokładny moment zgaśnięcia
```

**Iteracja 3:** Bardziej restrykcyjne progi
```python
# Wykrywaj tylko bardzo jasne światło (aktywny laser)
very_bright_pixels > 220 AND >= 1%  # Białe/żółte centrum
red_hot >= 3%  # Intensywna czerwień
```

**Wynik:** Dokładne wykrywanie końca spawania ✅

---

### Problem 6: Trim to motion zostawiał za dużo klatek na końcu

**Objawy:**
- Po zakończeniu ruchu nagranie trwa jeszcze 0.5s
- Statyczne klatki na końcu

**Przyczyna:**
- Padding na końcu segmentu = 30 klatek (0.5s przy 60 FPS)

**Rozwiązanie:**
```python
# Było:
seg_end = end + self.padding_frames  # +30 klatek

# Jest:
seg_end = end + 5  # Tylko 5 klatek (~0.08s)
```

Zachowujemy pełny padding (30 klatek) na **początku** ruchu (żeby złapać start), ale minimalny (5 klatek) na **końcu**.

**Wynik:** Precyzyjne ucięcie na końcu ruchu ✅

---

## 📁 Struktura projektu {#struktura}

```
welding-detector/
├── app/                          # Backend (Python/FastAPI)
│   ├── main.py                   # Entry point - uruchamia serwer
│   ├── config/
│   │   └── settings.py           # Konfiguracja (porty, FPS, rozdzielczość)
│   ├── api/
│   │   ├── routes.py             # Endpointy API (40+ endpointów)
│   │   └── models.py             # Modele Pydantic (Request/Response)
│   └── services/                 # Logika biznesowa
│       ├── camera_service.py               # Kamera USB + streaming + nagrywanie
│       ├── frame_overlay_service.py        # Timestamp + REC overlay
│       ├── video_overlay_service.py        # Overlay dla gotowych filmów
│       ├── motion_detection_service.py     # Wykrywanie ruchu + spawania
│       ├── frame_extractor_service.py      # Ekstrakcja klatek z MP4
│       ├── image_enhancement_service.py    # Filtry (CLAHE, sharpen, denoise)
│       ├── labeling_service.py             # Oznaczanie OK/NOK + wady
│       ├── ml_classification_service.py    # Model OK/NOK + trening
│       ├── defect_classifier_service.py    # Model typów wad + trening
│       └── video_analysis_service.py       # Batch analiza całych filmów
│
├── app_frontend/                 # Frontend (Vue.js 3)
│   ├── src/
│   │   ├── App.vue               # Główny komponent (1800+ linii)
│   │   ├── main.js               # Entry point Vue
│   │   └── style.css             # Style globalne + Tailwind
│   ├── index.html                # Szablon HTML
│   ├── package.json              # Zależności npm
│   └── vite.config.js            # Konfiguracja Vite
│
├── recordings/                   # Nagrania wideo
│   ├── *.mp4                     # Pliki wideo
│   └── analysis/                 # Wyniki analiz
│       └── *.json                # {filename: {summary, frames, defects}}
│
├── labels/                       # Dane treningowe
│   └── training_data/
│       ├── ok/                   # Zdjęcia OK
│       ├── nok/                  # Zdjęcia NOK
│       └── defects/              # Typy wad
│           ├── porosity/
│           ├── crack/
│           └── ...
│
├── models/                       # Wytrenowane modele AI
│   ├── latest_model.pth          # Model binarny OK/NOK
│   ├── training_info.json        # Metryki (accuracy, loss)
│   └── defects/
│       ├── defect_classifier.pth # Model 9-klasowy (typy wad)
│       └── training_info.json
│
├── requirements.txt              # Zależności Python
├── pytest.ini                    # Konfiguracja testów
├── Dockerfile                    # Backend Docker
├── docker-compose.yml            # Orchestracja (backend + frontend)
├── README.md                     # Oryginalny README
└── README2.md                    # Ten dokument 👈
```

---

## 🚀 Jak uruchomić aplikację {#uruchomienie}

### Wymagania

**Sprzęt:**
- Kamera USB (kompatybilna z Windows DirectShow/MSMF)
- CPU: Intel i5/Ryzen 5 lub lepszy
- RAM: 8GB minimum, 16GB zalecane
- (Opcjonalnie) GPU NVIDIA z CUDA dla szybszego treningu

**Oprogramowanie:**
- Windows 10/11
- Python 3.9+ (zalecane 3.11)
- Node.js 18+ (dla frontendu)
- Git

---

### Metoda 1: Ręczne uruchomienie

#### Backend (Terminal 1)

```bash
# 1. Sklonuj repozytorium
git clone <repo-url>
cd welding-detector

# 2. Utwórz wirtualne środowisko Python
python -m venv venv
venv\Scripts\activate  # Windows

# 3. Zainstaluj zależności
pip install -r requirements.txt

# 4. Uruchom serwer
python -m app.main
# Lub:
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# ✅ Backend działa na http://localhost:8000
# Dokumentacja API: http://localhost:8000/docs
```

#### Frontend (Terminal 2)

```bash
# 1. Przejdź do folderu frontendu
cd app_frontend

# 2. Zainstaluj zależności npm
npm install

# 3. Uruchom dev server
npm run dev

# ✅ Frontend działa na http://localhost:3000
```

#### Otwórz przeglądarkę

```
http://localhost:3000
```

Powinieneś zobaczyć:
- Live stream z kamery
- Przyciski nagrywania, analizy, etc.

---

### Metoda 2: Docker (jeśli skonfigurowane)

```bash
docker-compose up --build
```

---

## 🎓 Najważniejsze pojęcia wyjaśnione prostymi słowami

### 🤖 Sieć neuronowa (Neural Network)
Wyobraź sobie wiele warstw filtrów, które uczą się rozpoznawać wzorce na zdjęciach:
- Warstwa 1: wykrywa krawędzie
- Warstwa 2: wykrywa kształty (okrąg, linia)
- Warstwa 3: wykrywa tekstury (metal, pęknięcie)
- Warstwa N: "To jest pęknięcie spawu!"

### 📊 Trening modelu (Training)
Pokazujesz komputerowi 1000 zdjęć: "To jest OK, to jest NOK, to jest pęknięcie..."
Komputer dostosowuje parametry (miliony liczb), żeby się nauczyć rozpoznawać wzorce.

### 🔮 Przewidywanie (Inference)
Pokazujesz wytrenowanemu modelowi nowe zdjęcie → on mówi: "NOK z pewnością 92%".

### 🗺️ Grad-CAM (Gradient-weighted Class Activation Mapping)
Wizualizacja "na co patrzy AI":
- Czerwone obszary = tu model widzi wadę
- Niebieskie obszary = nieważne dla decyzji

### 🎞️ FPS (Frames Per Second)
Ile zdjęć (klatek) na sekundę:
- 30 FPS = 30 zdjęć/sekundę = płynny obraz
- 60 FPS = 60 zdjęć/sekundę = bardzo płynny
- Oko ludzkie widzi ~24 FPS jako płynny ruch

### 📦 MJPEG (Motion JPEG)
Format wideo gdzie każda klatka to osobny JPEG:
- Klatka 1: JPEG
- Klatka 2: JPEG
- Klatka 3: JPEG
- ...

Zalety: Szybkie, sprzętowe wsparcie w kamerach USB

### 🎥 MP4 (H.264/H.265)
Format wideo ze skompresją:
- Zapisuje pełną klatkę co kilka sekund (I-frame)
- Reszta to różnice (P-frames, B-frames)
- Mniejsze pliki niż MJPEG, ale wolniejsze przetwarzanie

### 🌊 Streaming
Ciągłe wysyłanie danych (wideo/audio) kawałek po kawałku:
```
Kamera → [klatka 1] → Przeglądarka (wyświetl)
       → [klatka 2] → Przeglądarka (wyświetl)
       → [klatka 3] → Przeglądarka (wyświetl)
       ...
```

### 🔄 Async/Await (Asynchroniczność)
Wielozadaniowość bez blokowania:
```python
async def pobierz_dane():
    await fetch(url)  # Czekaj, ale nie blokuj innych zadań
```

Jak kucharz gotujący kilka potraw naraz (nie czeka aż woda się zagotuje, lecz robi coś innego).

### 🎯 REST API
Sposób komunikacji frontend ↔ backend przez HTTP:
```
GET /camera/stream → pobierz stream
POST /recording/start → rozpocznij nagrywanie
GET /recording/list → pobierz listę nagrań
DELETE /recording/xyz.mp4 → usuń nagranie
```

### 📡 HTTP Request/Response
```
[Frontend]                    [Backend]
    |                             |
    |  GET /camera/health  --->  |
    |                             | (sprawdź kamerę)
    |  <--- 200 OK                |
    |  { "status": "healthy" }    |
```

---

## 🎉 Podsumowanie

**Welding Detector** to zaawansowany system kontroli jakości spawania, który łączy:

✅ **Monitoring w czasie rzeczywistym** (live stream z kamery)  
✅ **Nagrywanie wideo** z overlay timestamp  
✅ **Sztuczną inteligencję** (EfficientNet + PyTorch)  
✅ **Automatyczną analizę** całych nagrań  
✅ **Inteligentne przycinanie** (ruch, spawanie)  
✅ **Trenowanie własnych modeli** (transfer learning)  
✅ **Intuicyjny interfejs** (Vue.js + TailwindCSS)  

Projekt rozwija się iteracyjnie, rozwiązując realne problemy:
- Opóźnienie streamu
- Precyzyjne wykrywanie spawania
- Persystencja wyników
- Optymalizacja wydajności

**Technologie:**
- **Backend:** Python, FastAPI, OpenCV, PyTorch, NumPy
- **Frontend:** Vue.js 3, Vite, TailwindCSS
- **AI:** EfficientNet-B0, Grad-CAM, Transfer Learning
- **Video:** MJPEG streaming, MP4 encoding/decoding, motion detection

---

## 📞 Kontakt i rozwój

Aplikacja jest aktywnie rozwijana. Planowane funkcje:
- 📊 Dashboard ze statystykami długoterminowymi
- 📤 Export raportów (PDF, Excel)
- 🔔 Powiadomienia real-time (WebSockets)
- 🌐 Multi-camera support
- 🧠 Bardziej zaawansowane modele AI (YOLO, Segmentation)

---

**Autor:** welding-detector team  
**Wersja:** 1.0  
**Data:** Styczeń 2026  

---

*Ten dokument został stworzony z myślą o osobach nieznających się na programowaniu. Jeśli coś jest niejasne, zadaj pytanie!* 😊
