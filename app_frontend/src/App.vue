

<template>
  <div class="min-h-screen bg-gray-100 p-4 font-sans">
    <!-- Header -->
    <header class="flex justify-between items-center mb-4 p-4 bg-white rounded shadow">
      <h1 class="text-2xl font-bold">🎥 Welding Detector</h1>
      <div class="flex items-center gap-2">
        <span v-if="isRecording" class="text-red-500 font-mono font-bold">
          🔴 REC {{ formatDuration(recordingDuration) }}
        </span>
        <span v-else class="text-green-600">● Online</span>
      </div>
    </header>

    <!-- Stream (overlay z timestampem i REC jest w backendzie) -->
    <div class="bg-black mb-4 rounded shadow flex items-center justify-center overflow-hidden relative" style="height: 70vh;">
      <img 
        :src="streamUrl" 
        class="object-contain w-full h-full"
        alt="Live stream"
        @error="streamError = true"
        @load="streamError = false"
      />
      <span v-if="streamError" class="text-gray-500 absolute">❌ Brak połączenia z kamerą</span>
    </div>

    <!-- Buttons -->
    <div class="flex flex-wrap gap-3 mb-4">
      <button 
        @click="capture" 
        class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition"
      >
        📸 Capture
      </button>
      <button 
        @click="startRecording" 
        :disabled="isRecording"
        class="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600 transition disabled:opacity-50 disabled:cursor-not-allowed"
      >
        🔴 Start REC
      </button>
      <button 
        @click="stopRecording" 
        :disabled="!isRecording"
        class="bg-gray-500 text-white px-4 py-2 rounded hover:bg-gray-600 transition disabled:opacity-50 disabled:cursor-not-allowed"
      >
        ⏹️ Stop
      </button>
      <button 
        @click="fetchRecordings" 
        class="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 transition"
      >
        🔄 Refresh
      </button>
      <button 
        @click="showSettings = !showSettings" 
        class="bg-purple-500 text-white px-4 py-2 rounded hover:bg-purple-600 transition"
      >
        ⚙️ Ustawienia
      </button>
    </div>

    <!-- Camera Settings Panel -->
    <div v-if="showSettings" class="bg-white p-4 rounded shadow mb-4">
      <div class="flex justify-between items-center mb-4">
        <h2 class="text-xl font-semibold">⚙️ Ustawienia kamery</h2>
        <button @click="showSettings = false" class="text-gray-500 hover:text-gray-700">✕</button>
      </div>

      <!-- Ustawienia -->
      <div class="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
        <!-- Kontrast -->
        <div class="space-y-2">
          <label class="font-medium">🎚️ Kontrast</label>
          <input 
            type="range" 
            min="0" 
            max="255" 
            v-model.number="cameraSettings.contrast"
            @change="updateSetting('contrast', cameraSettings.contrast)"
            class="w-full"
          >
          <div class="text-xs text-gray-500 flex justify-between">
            <span>0</span>
            <span class="font-mono text-lg">{{ cameraSettings.contrast }}</span>
            <span>255</span>
          </div>
        </div>

        <!-- Jakość JPEG -->
        <div class="space-y-2">
          <label class="font-medium">🖼️ Jakość JPEG</label>
          <input 
            type="range" 
            min="50" 
            max="100" 
            v-model.number="cameraSettings.jpeg_quality"
            @change="updateSetting('jpeg_quality', cameraSettings.jpeg_quality)"
            class="w-full"
          >
          <div class="text-xs text-gray-500 flex justify-between">
            <span>50%</span>
            <span class="font-mono text-lg">{{ cameraSettings.jpeg_quality }}%</span>
            <span>100%</span>
          </div>
        </div>

        <!-- FPS -->
        <div class="space-y-2">
          <label class="font-medium">🎬 FPS</label>
          <select 
            v-model.number="cameraSettings.fps"
            @change="updateSetting('fps', cameraSettings.fps)"
            class="w-full p-2 border rounded text-lg"
          >
            <option :value="15">15 fps</option>
            <option :value="30">30 fps</option>
            <option :value="60">60 fps</option>
          </select>
        </div>

        <!-- Rozdzielczość -->
        <div class="space-y-2">
          <label class="font-medium">📐 Rozdzielczość</label>
          <select 
            v-model="cameraSettings.resolution"
            @change="updateSetting('resolution', cameraSettings.resolution)"
            class="w-full p-2 border rounded text-lg"
          >
            <option value="HD">HD (1280×720)</option>
            <option value="FHD">FHD (1920×1080)</option>
          </select>
        </div>

        <!-- Monochrom -->
        <div class="space-y-2">
          <label class="font-medium">🎨 Tryb obrazu</label>
          <button 
            @click="toggleMonochrome" 
            class="w-full p-2 rounded text-lg font-medium transition"
            :class="monochrome ? 'bg-gray-800 text-white' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'"
          >
            {{ monochrome ? '⬛ Mono' : '🌈 Kolor' }}
          </button>
        </div>
      </div>
    </div>

    <!-- Toast notification -->
    <div 
      v-if="toast.show"
      class="fixed bottom-5 right-5 px-6 py-3 rounded-lg shadow-lg text-white transition-opacity"
      :class="toast.type === 'success' ? 'bg-green-600' : 'bg-red-600'"
    >
      {{ toast.message }}
    </div>

    <!-- Recordings list -->
    <div class="bg-white p-4 rounded shadow">
      <h2 class="text-xl font-semibold mb-2">📁 Recordings:</h2>
      
      <div v-if="recordings.length === 0" class="text-gray-500 text-center py-4">
        Brak nagrań
      </div>
      
      <table v-else class="w-full">
        <thead>
          <tr class="text-left border-b">
            <th class="py-2">Plik</th>
            <th class="py-2">Rozmiar</th>
            <th class="py-2">Notatka</th>
            <th class="py-2 text-right">Akcje</th>
          </tr>
        </thead>
        <tbody>
          <tr 
            v-for="rec in recordings" 
            :key="rec.filename"
            class="border-b last:border-0 hover:bg-gray-50"
          >
            <td class="py-2">
              <span class="font-medium">🎬 {{ rec.filename }}</span>
              <span 
                v-if="rec.filename.includes('_trimmed')" 
                class="text-xs ml-2 px-2 py-0.5 rounded bg-green-200 text-green-800"
              >
                ✂️ Przycięte
              </span>
              <span 
                v-else-if="trimStatus[rec.filename] === 'trimming'" 
                class="text-xs ml-2 px-2 py-0.5 rounded bg-yellow-200 text-yellow-800 animate-pulse"
              >
                ⏳ Przycinanie...
              </span>
              <span 
                v-if="overlayStatus[rec.filename]" 
                class="text-xs ml-2 px-2 py-0.5 rounded"
                :class="{
                  'bg-yellow-200 text-yellow-800': overlayStatus[rec.filename].status === 'processing',
                  'bg-green-200 text-green-800': overlayStatus[rec.filename].status === 'completed',
                  'bg-red-200 text-red-800': overlayStatus[rec.filename].status === 'failed'
                }"
              >
                {{ overlayStatus[rec.filename].status === 'processing' 
                  ? `⏳ ${overlayStatus[rec.filename].progress || 0}%` 
                  : overlayStatus[rec.filename].status === 'completed' ? '✅ Overlay' : '❌ Błąd' }}
              </span>
            </td>
            <td class="py-2 text-gray-500 text-sm">{{ rec.size_mb }} MB</td>
            <td class="py-2">
              <input 
                type="text" 
                :value="rec.note || ''"
                @blur="saveNote(rec.filename, $event.target.value)"
                @keyup.enter="$event.target.blur()"
                placeholder="Dodaj notatkę..."
                class="w-full px-2 py-1 text-sm border rounded hover:border-blue-400 focus:border-blue-500 focus:outline-none"
              >
            </td>
            <td class="py-2 text-right">
              <div class="flex gap-1 justify-end">
                <button 
                  @click="openFrameViewer(rec.filename)" 
                  class="text-indigo-500 hover:text-indigo-700 px-2 py-1 text-sm"
                  title="Przeglądaj klatki z filtrami"
                >
                  🔍
                </button>
                <button 
                  v-if="!rec.filename.includes('_trimmed') && trimStatus[rec.filename] !== 'trimming'"
                  @click="trimToMotion(rec.filename)" 
                  class="text-orange-500 hover:text-orange-700 px-2 py-1 text-sm"
                  title="Przytnij do ruchu"
                >
                  ✂️
                </button>
                <span 
                  v-else-if="trimStatus[rec.filename] === 'trimming'"
                  class="text-orange-400 px-2 py-1 text-sm animate-spin"
                >
                  ⏳
                </span>
                <button 
                  v-if="!rec.filename.includes('_overlay') && !overlayStatus[rec.filename]"
                  @click="applyOverlay(rec.filename)" 
                  class="text-purple-500 hover:text-purple-700 px-2 py-1 text-sm"
                  title="Nałóż timestamp"
                >
                  🎨
                </button>
                <button 
                  @click="downloadRecording(rec.filename)" 
                  class="text-blue-500 hover:text-blue-700 px-2 py-1"
                  title="Pobierz"
                >
                  ⬇️
                </button>
                <button 
                  @click="deleteRecording(rec.filename)" 
                  class="text-red-500 hover:text-red-700 px-2 py-1"
                  title="Usuń"
                >
                  🗑️
                </button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Frame Viewer Modal -->
    <div 
      v-if="frameViewer.show" 
      class="fixed inset-0 bg-black bg-opacity-75 z-50 flex items-center justify-center p-4"
      @click.self="frameViewer.show = false"
    >
      <div class="bg-white rounded-lg shadow-2xl w-full max-w-6xl max-h-[95vh] overflow-hidden flex flex-col">
        <!-- Header -->
        <div class="flex justify-between items-center p-4 border-b bg-gray-50">
          <h2 class="text-xl font-bold">🔍 {{ frameViewer.filename }} - Klatka {{ frameViewer.currentFrame }}/{{ frameViewer.totalFrames - 1 }}</h2>
          <button @click="frameViewer.show = false" class="text-gray-500 hover:text-gray-700 text-2xl">✕</button>
        </div>
        
        <!-- Content -->
        <div class="flex flex-1 overflow-hidden">
          <!-- Image -->
          <div class="flex-1 bg-gray-900 flex items-center justify-center p-4 relative">
            <img 
              :src="frameViewer.imageUrl" 
              class="max-w-full max-h-full object-contain"
              :class="{ 'opacity-50': frameViewer.loading }"
            >
            <div v-if="frameViewer.loading" class="absolute inset-0 flex items-center justify-center">
              <span class="text-white text-4xl animate-spin">⏳</span>
            </div>
          </div>
          
          <!-- Sidebar - Filters -->
          <div class="w-80 border-l bg-gray-50 p-4 overflow-y-auto">
            <h3 class="font-bold text-lg mb-4">🎨 Filtry obrazu</h3>
            
            <!-- Preset -->
            <div class="mb-4">
              <label class="font-medium text-sm">Preset</label>
              <select 
                v-model="frameViewer.filters.preset" 
                @change="updateFrameImage"
                class="w-full mt-1 p-2 border rounded"
              >
                <option value="">-- Brak --</option>
                <option value="weld_enhance">🔧 Weld Enhance (spawy)</option>
                <option value="high_contrast">⚡ High Contrast</option>
                <option value="edge_overlay">🔲 Edge Overlay</option>
                <option value="heatmap">🌡️ Heatmap</option>
                <option value="denoise">🔇 Denoise</option>
              </select>
            </div>
            
            <hr class="my-4">
            <h4 class="font-medium text-sm mb-3">Ręczne ustawienia</h4>
            
            <!-- CLAHE -->
            <div class="mb-3">
              <label class="text-sm flex justify-between">
                <span>CLAHE (kontrast lokalny)</span>
                <span class="font-mono">{{ frameViewer.filters.clahe || 'OFF' }}</span>
              </label>
              <input type="range" min="0" max="4" step="0.5" 
                v-model.number="frameViewer.filters.clahe" 
                @change="updateFrameImage"
                class="w-full">
            </div>
            
            <!-- Sharpen -->
            <div class="mb-3">
              <label class="text-sm flex justify-between">
                <span>Sharpen (ostrość)</span>
                <span class="font-mono">{{ frameViewer.filters.sharpen || 'OFF' }}</span>
              </label>
              <input type="range" min="0" max="3" step="0.5" 
                v-model.number="frameViewer.filters.sharpen" 
                @change="updateFrameImage"
                class="w-full">
            </div>
            
            <!-- Gamma -->
            <div class="mb-3">
              <label class="text-sm flex justify-between">
                <span>Gamma (jasność)</span>
                <span class="font-mono">{{ frameViewer.filters.gamma.toFixed(1) }}</span>
              </label>
              <input type="range" min="0.3" max="3" step="0.1" 
                v-model.number="frameViewer.filters.gamma" 
                @change="updateFrameImage"
                class="w-full">
            </div>
            
            <!-- Contrast -->
            <div class="mb-3">
              <label class="text-sm flex justify-between">
                <span>Contrast</span>
                <span class="font-mono">{{ frameViewer.filters.contrast.toFixed(1) }}</span>
              </label>
              <input type="range" min="0.5" max="3" step="0.1" 
                v-model.number="frameViewer.filters.contrast" 
                @change="updateFrameImage"
                class="w-full">
            </div>
            
            <!-- Denoise -->
            <div class="mb-3">
              <label class="text-sm flex justify-between">
                <span>Denoise</span>
                <span class="font-mono">{{ frameViewer.filters.denoise || 'OFF' }}</span>
              </label>
              <input type="range" min="0" max="15" step="1" 
                v-model.number="frameViewer.filters.denoise" 
                @change="updateFrameImage"
                class="w-full">
            </div>
            
            <!-- Edges -->
            <div class="mb-3">
              <label class="flex items-center gap-2 text-sm">
                <input type="checkbox" v-model="frameViewer.filters.edges" @change="updateFrameImage">
                <span>🔲 Edge overlay</span>
              </label>
            </div>
            
            <!-- Heatmap -->
            <div class="mb-3">
              <label class="text-sm">Heatmap</label>
              <select 
                v-model="frameViewer.filters.heatmap" 
                @change="updateFrameImage"
                class="w-full mt-1 p-2 border rounded text-sm"
              >
                <option value="">OFF</option>
                <option value="jet">🌈 Jet</option>
                <option value="hot">🔥 Hot</option>
                <option value="turbo">🌀 Turbo</option>
                <option value="viridis">🌿 Viridis</option>
                <option value="plasma">💜 Plasma</option>
              </select>
            </div>
            
            <!-- Reset -->
            <button 
              @click="resetFilters" 
              class="w-full mt-4 px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded text-sm"
            >
              🔄 Reset filtrów
            </button>
            
            <!-- Download -->
            <button 
              @click="downloadCurrentFrame" 
              class="w-full mt-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded text-sm"
            >
              💾 Pobierz klatkę
            </button>
          </div>
        </div>
        
        <!-- Footer - Navigation -->
        <div class="p-4 border-t bg-gray-50 flex items-center justify-between">
          <button 
            @click="prevFrame" 
            :disabled="frameViewer.currentFrame <= 0"
            class="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded disabled:opacity-50"
          >
            ⬅️ Poprzednia
          </button>
          
          <div class="flex items-center gap-2">
            <input 
              type="range" 
              :min="0" 
              :max="frameViewer.totalFrames - 1" 
              v-model.number="frameViewer.currentFrame"
              @change="updateFrameImage"
              class="w-64"
            >
            <input 
              type="number" 
              v-model.number="frameViewer.currentFrame"
              @change="updateFrameImage"
              :min="0" 
              :max="frameViewer.totalFrames - 1"
              class="w-20 p-1 border rounded text-center"
            >
          </div>
          
          <button 
            @click="nextFrame" 
            :disabled="frameViewer.currentFrame >= frameViewer.totalFrames - 1"
            class="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded disabled:opacity-50"
          >
            Następna ➡️
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

// API base URL - pusty bo używamy proxy Vite (działa w Docker i lokalnie)
const API = ''

// State
const isRecording = ref(false)
const recordingDuration = ref(0)
const recordings = ref([])
const overlayStatus = ref({})  // filename -> status
const trimStatus = ref({})  // filename -> 'trimming' | 'trimmed'
const streamError = ref(false)

// Frame Viewer state
const frameViewer = ref({
  show: false,
  filename: '',
  currentFrame: 0,
  totalFrames: 0,
  imageUrl: '',
  loading: false,
  filters: {
    preset: '',
    clahe: 0,
    sharpen: 0,
    gamma: 1.0,
    contrast: 1.0,
    denoise: 0,
    edges: false,
    heatmap: ''
  }
})
const streamUrl = ref(`/camera/stream`)  // Domyślnie płynny stream
const toast = ref({ show: false, message: '', type: 'success' })
const showSettings = ref(false)

// Camera settings - tylko działające
const cameraSettings = ref({
  contrast: 128,
  jpeg_quality: 90,
  fps: 30,
  resolution: 'HD'
})
const monochrome = ref(false)

let statusInterval = null
let overlayPollInterval = null

// Toast helper
function showToast(message, type = 'success') {
  toast.value = { show: true, message, type }
  setTimeout(() => toast.value.show = false, 3000)
}

// Format duration as MM:SS
function formatDuration(seconds) {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
}

// ============== API CALLS ==============

async function capture() {
  try {
    const response = await fetch(`${API}/camera/capture?overlay=true`)
    if (!response.ok) throw new Error('Błąd pobierania')
    
    const blob = await response.blob()
    const url = URL.createObjectURL(blob)
    
    const a = document.createElement('a')
    a.href = url
    a.download = `capture_${Date.now()}.jpg`
    a.click()
    URL.revokeObjectURL(url)
    
    showToast('📸 Zdjęcie zapisane')
  } catch (e) {
    showToast('❌ ' + e.message, 'error')
  }
}

async function startRecording() {
  try {
    const response = await fetch(`${API}/recording/start`, { method: 'POST' })
    if (!response.ok) throw new Error('Nie można rozpocząć nagrywania')
    
    isRecording.value = true
    showToast('🔴 Nagrywanie rozpoczęte')
  } catch (e) {
    showToast('❌ ' + e.message, 'error')
  }
}

async function stopRecording() {
  try {
    const response = await fetch(`${API}/recording/stop`, { method: 'POST' })
    if (!response.ok) throw new Error('Nie można zatrzymać')
    
    const data = await response.json()
    
    isRecording.value = false
    recordingDuration.value = 0
    
    showToast(`⏹️ Zapisano: ${data.filename} (${data.duration_seconds}s)`)
    fetchRecordings()
  } catch (e) {
    showToast('❌ ' + e.message, 'error')
  }
}

async function fetchRecordings() {
  try {
    const response = await fetch(`${API}/recording/list`)
    const data = await response.json()
    recordings.value = data.recordings || []
  } catch (e) {
    console.error('Error fetching recordings:', e)
    showToast('❌ Nie można pobrać listy nagrań', 'error')
  }
}

async function saveNote(filename, note) {
  try {
    const response = await fetch(`${API}/recording/${filename}/note?note=${encodeURIComponent(note)}`, { method: 'PUT' })
    if (!response.ok) throw new Error('Błąd zapisu')
    
    // Aktualizuj lokalnie
    const rec = recordings.value.find(r => r.filename === filename)
    if (rec) rec.note = note
  } catch (e) {
    showToast('❌ Nie udało się zapisać notatki', 'error')
  }
}

function downloadRecording(filename) {
  window.open(`${API}/recording/download/${filename}`)
}

async function deleteRecording(filename) {
  if (!confirm(`Usunąć ${filename}?`)) return
  
  try {
    const response = await fetch(`${API}/recording/${filename}`, { method: 'DELETE' })
    if (!response.ok) throw new Error('Nie można usunąć')
    
    showToast('🗑️ Usunięto')
    fetchRecordings()
  } catch (e) {
    showToast('❌ ' + e.message, 'error')
  }
}

async function applyOverlay(filename) {
  try {
    const response = await fetch(`${API}/recording/${filename}/apply-overlay`, { method: 'POST' })
    if (!response.ok) throw new Error('Nie można rozpocząć przetwarzania')
    
    overlayStatus.value[filename] = { status: 'processing', progress: 0 }
    showToast('🎨 Nakładanie overlay rozpoczęte')
    
    // Rozpocznij polling statusu
    startOverlayPolling()
  } catch (e) {
    showToast('❌ ' + e.message, 'error')
  }
}

async function trimToMotion(filename) {
  try {
    trimStatus.value[filename] = 'trimming'
    showToast('✂️ Przycinanie do ruchu rozpoczęte...')
    
    const response = await fetch(`${API}/recording/${filename}/trim-to-motion`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    })
    
    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Nie można przyciąć wideo')
    }
    
    const data = await response.json()
    
    if (data.status === 'no_motion') {
      showToast('⚠️ Nie wykryto ruchu w nagraniu', 'error')
      delete trimStatus.value[filename]
    } else {
      showToast(`✂️ Przycięto! ${data.output_filename} (${data.duration_seconds}s, -${data.reduction_percent}%)`)
      delete trimStatus.value[filename]
      fetchRecordings()
    }
  } catch (e) {
    showToast('❌ ' + e.message, 'error')
    delete trimStatus.value[filename]
  }
}

async function pollOverlayStatus() {
  try {
    const response = await fetch(`${API}/recording/overlay-jobs`)
    const data = await response.json()
    
    // Aktualizuj statusy
    for (const [filename, status] of Object.entries(data)) {
      overlayStatus.value[filename] = status
      
      // Jeśli zakończone, odśwież listę nagrań
      if (status.status === 'completed') {
        fetchRecordings()
      }
    }
    
    // Jeśli nie ma aktywnych zadań, zatrzymaj polling
    const hasActive = Object.values(data).some(s => s.status === 'processing')
    if (!hasActive && overlayPollInterval) {
      clearInterval(overlayPollInterval)
      overlayPollInterval = null
    }
  } catch (e) {
    console.error('Overlay status check failed:', e)
  }
}

function startOverlayPolling() {
  if (overlayPollInterval) return
  overlayPollInterval = setInterval(pollOverlayStatus, 2000)
}

async function pollRecordingStatus() {
  try {
    const response = await fetch(`${API}/recording/status`)
    const data = await response.json()
    
    isRecording.value = data.is_recording
    recordingDuration.value = data.duration_seconds ? Math.floor(data.duration_seconds) : 0
  } catch (e) {
    console.error('Status check failed:', e)
  }
}

// ============== CAMERA SETTINGS ==============

async function fetchCameraSettings() {
  try {
    const response = await fetch(`${API}/camera/settings`)
    const data = await response.json()
    
    // Aktualizuj lokalne ustawienia
    if (data.contrast !== undefined) cameraSettings.value.contrast = Math.round(data.contrast)
    if (data.jpeg_quality !== undefined) cameraSettings.value.jpeg_quality = data.jpeg_quality
    if (data.fps !== undefined) cameraSettings.value.fps = Math.round(data.fps)
    if (data.resolution !== undefined) cameraSettings.value.resolution = data.resolution
    
  } catch (e) {
    console.error('Failed to fetch camera settings:', e)
  }
}

async function updateSetting(name, value) {
  try {
    const body = {}
    body[name] = value
    
    const response = await fetch(`${API}/camera/settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })
    
    if (!response.ok) throw new Error('Błąd aktualizacji')
    showToast(`✅ ${name} = ${value}`)
  } catch (e) {
    showToast('❌ ' + e.message, 'error')
  }
}

async function toggleMonochrome() {
  try {
    const newValue = !monochrome.value
    const response = await fetch(`${API}/camera/monochrome?enabled=${newValue}`, { method: 'POST' })
    if (!response.ok) throw new Error('Błąd przełączania')
    
    const data = await response.json()
    monochrome.value = data.monochrome
    showToast(monochrome.value ? '⬛ Tryb mono' : '🌈 Tryb kolorowy')
  } catch (e) {
    showToast('❌ ' + e.message, 'error')
  }
}

async function fetchMonochrome() {
  try {
    const response = await fetch(`${API}/camera/monochrome`)
    const data = await response.json()
    monochrome.value = data.monochrome
  } catch (e) {
    console.error('Failed to fetch monochrome:', e)
  }
}

// ============== FRAME VIEWER ==============

async function openFrameViewer(filename) {
  frameViewer.value.filename = filename
  frameViewer.value.currentFrame = 0
  frameViewer.value.loading = true
  frameViewer.value.show = true
  resetFilters()
  
  try {
    const response = await fetch(`${API}/recording/${filename}/info`)
    if (!response.ok) throw new Error('Nie można pobrać info o wideo')
    const info = await response.json()
    frameViewer.value.totalFrames = info.frame_count
    updateFrameImage()
  } catch (e) {
    showToast('❌ ' + e.message, 'error')
    frameViewer.value.show = false
  }
}

function updateFrameImage() {
  frameViewer.value.loading = true
  
  const f = frameViewer.value.filters
  const params = new URLSearchParams()
  
  if (f.preset) params.append('preset', f.preset)
  if (f.clahe > 0) params.append('clahe', f.clahe)
  if (f.sharpen > 0) params.append('sharpen', f.sharpen)
  if (f.gamma !== 1.0) params.append('gamma', f.gamma)
  if (f.contrast !== 1.0) params.append('contrast', f.contrast)
  if (f.denoise > 0) params.append('denoise', f.denoise)
  if (f.edges) params.append('edges', 'true')
  if (f.heatmap) params.append('heatmap', f.heatmap)
  
  const queryString = params.toString()
  const url = `${API}/recording/${frameViewer.value.filename}/frame/${frameViewer.value.currentFrame}${queryString ? '?' + queryString : ''}`
  
  frameViewer.value.imageUrl = url + (queryString ? '&' : '?') + '_t=' + Date.now()
  
  const img = new Image()
  img.onload = () => frameViewer.value.loading = false
  img.onerror = () => frameViewer.value.loading = false
  img.src = frameViewer.value.imageUrl
}

function resetFilters() {
  frameViewer.value.filters = {
    preset: '',
    clahe: 0,
    sharpen: 0,
    gamma: 1.0,
    contrast: 1.0,
    denoise: 0,
    edges: false,
    heatmap: ''
  }
  if (frameViewer.value.show) updateFrameImage()
}

function prevFrame() {
  if (frameViewer.value.currentFrame > 0) {
    frameViewer.value.currentFrame--
    updateFrameImage()
  }
}

function nextFrame() {
  if (frameViewer.value.currentFrame < frameViewer.value.totalFrames - 1) {
    frameViewer.value.currentFrame++
    updateFrameImage()
  }
}

function downloadCurrentFrame() {
  const a = document.createElement('a')
  a.href = frameViewer.value.imageUrl
  a.download = `${frameViewer.value.filename}_frame${frameViewer.value.currentFrame}.jpg`
  a.click()
}

// Lifecycle
onMounted(() => {
  fetchRecordings()
  pollRecordingStatus()
  fetchCameraSettings()
  fetchMonochrome()
  
  // Polling co 2s gdy nagrywamy, co 5s gdy nie
  statusInterval = setInterval(() => {
    pollRecordingStatus()
  }, isRecording.value ? 2000 : 5000)
})

onUnmounted(() => {
  if (statusInterval) clearInterval(statusInterval)
  if (overlayPollInterval) clearInterval(overlayPollInterval)
})
</script>

<style scoped>
/* Tailwind handles everything */
</style>
