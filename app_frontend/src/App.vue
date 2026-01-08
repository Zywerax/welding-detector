

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
              
              <!-- Analysis status badge -->
              <span 
                v-if="rec.analysis"
                class="text-xs ml-2 px-2 py-0.5 rounded"
                :class="{
                  'bg-blue-200 text-blue-800': rec.analysis.in_progress,
                  'bg-green-200 text-green-800': rec.analysis.results && !rec.analysis.in_progress,
                  'bg-red-200 text-red-800': rec.analysis.error
                }"
                :title="getAnalysisSummary(rec)"
              >
                {{ rec.analysis.in_progress 
                  ? `🔍 ${rec.analysis.progress}%` 
                  : rec.analysis.results 
                    ? `✅ OK:${rec.analysis.results.summary.ok} NOK:${rec.analysis.results.summary.nok}`
                    : '❌ Błąd' }}
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
                  @click="startVideoAnalysis(rec.filename)" 
                  :disabled="rec.analysis?.in_progress"
                  class="px-2 py-1 text-sm"
                  :class="rec.analysis?.in_progress ? 'text-gray-400 cursor-not-allowed' : 'text-blue-500 hover:text-blue-700'"
                  title="Analizuj wideo"
                >
                  🔬
                </button>
                <button 
                  v-if="rec.analysis?.results && !rec.analysis.in_progress"
                  @click="viewAnalysisResults(rec.filename)" 
                  class="text-green-500 hover:text-green-700 px-2 py-1 text-sm"
                  title="Zobacz wyniki analizy"
                >
                  📊
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

            <!-- Labeling Section -->
            <hr class="my-4">
            <h4 class="font-bold text-sm mb-3">🏷️ Etykietowanie</h4>
            
            <!-- Current label display -->
            <div v-if="frameViewer.currentLabel" class="mb-3 p-2 rounded text-center text-sm font-bold"
              :class="{
                'bg-green-200 text-green-800': frameViewer.currentLabel === 'ok',
                'bg-red-200 text-red-800': frameViewer.currentLabel === 'nok',
                'bg-gray-200 text-gray-600': frameViewer.currentLabel === 'skip'
              }">
              {{ frameViewer.currentLabel === 'ok' ? '✅ OK' : frameViewer.currentLabel === 'nok' ? '❌ NOK' : '⏭️ SKIP' }}
            </div>
            
            <!-- Label buttons -->
            <div class="grid grid-cols-3 gap-2 mb-3">
              <button 
                @click="labelFrame('ok')"
                class="px-3 py-3 bg-green-500 hover:bg-green-600 text-white rounded font-bold text-lg"
                :class="{ 'ring-4 ring-green-300': frameViewer.currentLabel === 'ok' }"
              >
                ✅ OK
              </button>
              <button 
                @click="showDefectSelector = true"
                class="px-3 py-3 bg-red-500 hover:bg-red-600 text-white rounded font-bold text-lg"
                :class="{ 'ring-4 ring-red-300': frameViewer.currentLabel === 'nok' }"
              >
                ❌ NOK
              </button>
              <button 
                @click="labelFrame('skip')"
                class="px-3 py-3 bg-gray-400 hover:bg-gray-500 text-white rounded font-bold text-lg"
                :class="{ 'ring-4 ring-gray-300': frameViewer.currentLabel === 'skip' }"
              >
                ⏭️
              </button>
            </div>
            
            <!-- Defect type selector (shown after clicking NOK) -->
            <div v-if="showDefectSelector" class="mb-3 p-3 bg-red-50 rounded border-2 border-red-200">
              <h5 class="font-bold text-sm mb-2 text-red-800">🔍 Wybierz typ wady:</h5>
              <div class="grid grid-cols-2 gap-1">
                <button 
                  v-for="defect in defectTypes" :key="defect.value"
                  @click="labelFrameWithDefect(defect.value)"
                  class="px-2 py-2 bg-red-100 hover:bg-red-200 text-red-800 rounded text-xs font-medium text-left"
                >
                  {{ defect.icon }} {{ defect.label }}
                </button>
              </div>
              <button 
                @click="showDefectSelector = false"
                class="w-full mt-2 px-2 py-1 bg-gray-300 hover:bg-gray-400 text-gray-700 rounded text-xs"
              >
                ❌ Anuluj
              </button>
            </div>
            
            <!-- Current defect type display -->
            <div v-if="frameViewer.currentDefectType && frameViewer.currentLabel === 'nok'" 
              class="mb-3 p-2 bg-red-100 rounded text-center text-sm">
              <span class="text-red-800">Typ wady: <strong>{{ getDefectLabel(frameViewer.currentDefectType) }}</strong></span>
            </div>
            
            <!-- Auto-advance -->
            <label class="flex items-center gap-2 text-sm mb-3">
              <input type="checkbox" v-model="frameViewer.autoAdvance">
              <span>Auto-przejdź do następnej</span>
            </label>
            
            <!-- Stats -->
            <div v-if="labelingStats" class="text-xs text-gray-500 bg-white p-2 rounded">
              <div class="flex justify-between">
                <span>✅ OK:</span>
                <span class="font-mono">{{ labelingStats.ok_count }}</span>
              </div>
              <div class="flex justify-between">
                <span>❌ NOK:</span>
                <span class="font-mono">{{ labelingStats.nok_count }}</span>
              </div>
              
              <!-- Defect types breakdown -->
              <div v-if="labelingStats.defect_counts && Object.keys(labelingStats.defect_counts).length > 0" 
                class="mt-2 pt-2 border-t border-gray-200">
                <div class="text-gray-600 mb-1 font-medium">Typy wad:</div>
                <div v-for="(count, type) in labelingStats.defect_counts" :key="type" 
                  class="flex justify-between text-gray-500 pl-2">
                  <span>{{ getDefectLabel(type) }}</span>
                  <span class="font-mono">{{ count }}</span>
                </div>
              </div>
              
              <div class="flex justify-between font-bold border-t mt-1 pt-1">
                <span>Razem:</span>
                <span class="font-mono">{{ labelingStats.total_labeled }}</span>
              </div>
              <div v-if="labelingStats.ok_count >= 20 && labelingStats.nok_count >= 20" 
                class="mt-2 text-green-600 font-bold text-center">
                🎉 Gotowe do treningu!
              </div>
            </div>
            
            <!-- ML Section -->
            <hr class="my-4">
            <h4 class="font-bold text-sm mb-3">🤖 AI Klasyfikacja</h4>
            
            <!-- Prediction result -->
            <div v-if="mlPrediction" class="mb-3 p-2 rounded text-center"
              :class="{
                'bg-green-200 text-green-800': mlPrediction.prediction === 'ok',
                'bg-red-200 text-red-800': mlPrediction.prediction === 'nok'
              }">
              <div class="font-bold text-lg">
                {{ mlPrediction.prediction === 'ok' ? '✅ OK' : '❌ NOK' }}
              </div>
              <div class="text-sm">
                Pewność: {{ mlPrediction.confidence }}%
              </div>
            </div>
            
            <!-- ML buttons -->
            <div class="space-y-2">
              <button 
                @click="predictFrame"
                :disabled="!mlInfo?.model_loaded || mlPredicting"
                class="w-full px-3 py-2 bg-indigo-500 hover:bg-indigo-600 text-white rounded text-sm disabled:opacity-50"
              >
                {{ mlPredicting ? '⏳ Analizuję...' : '🔍 Klasyfikuj AI' }}
              </button>
              
              <button 
                @click="showGradCAM"
                :disabled="!mlInfo?.model_loaded || !mlInfo?.gradcam_available"
                class="w-full px-3 py-2 bg-orange-500 hover:bg-orange-600 text-white rounded text-sm disabled:opacity-50"
              >
                🔥 Pokaż Grad-CAM
              </button>
            </div>
            
            <!-- ML model info -->
            <div v-if="mlInfo" class="mt-3 text-xs text-gray-500 bg-white p-2 rounded">
              <div class="flex justify-between">
                <span>Model:</span>
                <span :class="mlInfo.model_loaded ? 'text-green-600' : 'text-red-600'">
                  {{ mlInfo.model_loaded ? '✅ Załadowany' : '❌ Brak' }}
                </span>
              </div>
              <div v-if="mlInfo.training_data_stats" class="flex justify-between">
                <span>Dane treningowe:</span>
                <span>{{ mlInfo.training_data_stats.total_samples }}</span>
              </div>
              <button 
                v-if="mlInfo.training_data_stats?.ready_for_training && !mlInfo.model_loaded"
                @click="startTraining"
                :disabled="trainingInProgress"
                class="w-full mt-2 px-2 py-1 bg-purple-500 hover:bg-purple-600 text-white rounded text-xs"
              >
                {{ trainingInProgress ? '⏳ Trening...' : '🚀 Trenuj OK/NOK' }}
              </button>
              
              <!-- Defect classifier button -->
              <button 
                v-if="labelingStats && labelingStats.nok_count >= 10"
                @click="startDefectTraining"
                :disabled="defectTrainingInProgress"
                class="w-full mt-2 px-2 py-1 bg-orange-500 hover:bg-orange-600 text-white rounded text-xs"
              >
                {{ defectTrainingInProgress ? '⏳ Trening defektów...' : '🔥 Trenuj klasyfikator wad' }}
              </button>
            </div>
            
            <!-- Defect Classification Section -->
            <hr class="my-4">
            <h4 class="font-bold text-sm mb-3">🔍 Klasyfikacja Wad</h4>
            
            <!-- Defect prediction result -->
            <div v-if="defectPrediction" class="mb-3 p-2 bg-red-50 rounded border border-red-200">
              <div class="font-bold text-center text-red-800 mb-2">
                {{ getDefectLabel(defectPrediction.prediction) }}
              </div>
              <div class="text-xs text-center text-red-600 mb-2">
                Pewność: {{ defectPrediction.confidence }}%
              </div>
              <!-- Top 3 probabilities -->
              <div v-if="defectPrediction.class_probabilities" class="text-xs space-y-1">
                <div v-for="(prob, className) in getTopDefectProbabilities(defectPrediction.class_probabilities, 3)" 
                  :key="className"
                  class="flex justify-between items-center">
                  <span>{{ getDefectLabel(className) }}</span>
                  <span class="font-mono">{{ prob.toFixed(1) }}%</span>
                </div>
              </div>
            </div>
            
            <!-- Defect classifier buttons -->
            <div v-if="defectInfo?.model_loaded" class="space-y-2">
              <button 
                @click="predictDefect"
                :disabled="defectPredicting"
                class="w-full px-3 py-2 bg-red-500 hover:bg-red-600 text-white rounded text-sm disabled:opacity-50"
              >
                {{ defectPredicting ? '⏳ Analizuję...' : '🔍 Klasyfikuj wadę' }}
              </button>
              
              <button 
                @click="showDefectGradCAM"
                :disabled="!defectInfo?.gradcam_available"
                class="w-full px-3 py-2 bg-orange-500 hover:bg-orange-600 text-white rounded text-sm disabled:opacity-50"
              >
                🔥 Grad-CAM wady
              </button>
            </div>
            
            <!-- Defect model info -->
            <div v-if="defectInfo" class="mt-3 text-xs text-gray-500 bg-white p-2 rounded">
              <div class="flex justify-between">
                <span>Model wad:</span>
                <span :class="defectInfo.model_loaded ? 'text-green-600' : 'text-red-600'">
                  {{ defectInfo.model_loaded ? '✅ Załadowany' : '❌ Brak' }}
                </span>
              </div>
              <div v-if="defectInfo.training_data_stats" class="flex justify-between">
                <span>Klasy:</span>
                <span>{{ defectInfo.training_data_stats.num_classes || 0 }}</span>
              </div>
            </div>
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

  <!-- Analysis Results Modal -->
  <div 
    v-if="analysisResults.show" 
    class="fixed inset-0 bg-black bg-opacity-75 z-50 flex items-center justify-center p-4"
    @click.self="analysisResults.show = false"
  >
    <div class="bg-white rounded-lg shadow-2xl w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
      <!-- Header -->
      <div class="flex justify-between items-center p-4 border-b bg-gray-50">
        <h2 class="text-xl font-bold">📊 Wyniki analizy: {{ analysisResults.filename }}</h2>
        <button 
          @click="analysisResults.show = false"
          class="text-gray-600 hover:text-gray-800 text-2xl leading-none"
        >
          ×
        </button>
      </div>

      <!-- Results Content -->
      <div v-if="analysisResults.results" class="p-6 overflow-y-auto flex-1">
        <!-- Summary -->
        <div class="mb-6">
          <h3 class="text-lg font-semibold mb-3">Podsumowanie</h3>
          <div class="grid grid-cols-2 gap-4">
            <div class="bg-green-100 border border-green-300 rounded-lg p-4">
              <div class="text-3xl font-bold text-green-700">
                {{ analysisResults.results.summary.ok }}
              </div>
              <div class="text-sm text-green-600">Klatki OK ✅</div>
            </div>
            <div class="bg-red-100 border border-red-300 rounded-lg p-4">
              <div class="text-3xl font-bold text-red-700">
                {{ analysisResults.results.summary.nok }}
              </div>
              <div class="text-sm text-red-600">Klatki NOK ❌</div>
            </div>
          </div>
        </div>

        <!-- Defect Summary -->
        <div v-if="analysisResults.results.defect_summary && Object.keys(analysisResults.results.defect_summary).length > 0" class="mb-6">
          <h3 class="text-lg font-semibold mb-3">Wykryte wady</h3>
          <div class="grid grid-cols-2 gap-3">
            <div 
              v-for="(count, defectType) in analysisResults.results.defect_summary" 
              :key="defectType"
              class="bg-orange-100 border border-orange-300 rounded-lg p-3 flex items-center justify-between"
            >
              <span class="font-medium">
                {{ defectTypes.find(d => d.value === defectType)?.icon || '❓' }}
                {{ defectTypes.find(d => d.value === defectType)?.label || defectType }}
              </span>
              <span class="text-xl font-bold text-orange-700">{{ count }}</span>
            </div>
          </div>
        </div>

        <!-- Frame List -->
        <div>
          <h3 class="text-lg font-semibold mb-3">Szczegóły klatek NOK ({{ analysisResults.results.frames.filter(f => f.prediction === 'nok').length }})</h3>
          <div class="space-y-2 max-h-96 overflow-y-auto">
            <div 
              v-for="frame in analysisResults.results.frames.filter(f => f.prediction === 'nok')" 
              :key="frame.frame_number"
              class="bg-gray-50 border rounded p-3 hover:bg-gray-100"
            >
              <div class="flex gap-3">
                <!-- Thumbnail -->
                <div class="flex-shrink-0">
                  <img 
                    :src="`${API}/frames/${analysisResults.filename}/frame/${frame.frame}?size=thumbnail`"
                    :alt="`Frame ${frame.frame}`"
                    class="w-32 h-24 object-cover rounded border-2 border-red-300 cursor-pointer hover:border-red-500"
                    @click="openFrameInViewer(analysisResults.filename, frame.frame)"
                    title="Kliknij aby otworzyć w przeglądarce klatek"
                  />
                </div>
                
                <!-- Frame info -->
                <div class="flex-1 flex flex-col justify-between">
                  <div>
                    <div class="font-mono font-semibold text-lg">Klatka {{ frame.frame }}</div>
                    <div v-if="frame.defect_type" class="mt-1">
                      <span class="text-base">
                        {{ defectTypes.find(d => d.value === frame.defect_type)?.icon || '❓' }}
                        {{ defectTypes.find(d => d.value === frame.defect_type)?.label || frame.defect_type }}
                      </span>
                      <span class="text-sm text-gray-500 ml-2">({{ frame.defect_confidence?.toFixed(1) }}%)</span>
                    </div>
                  </div>
                  <div class="text-sm">
                    <span class="px-2 py-1 rounded bg-red-200 text-red-800">
                      NOK {{ frame.confidence?.toFixed(1) }}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Footer -->
      <div class="border-t p-4 bg-gray-50 flex justify-end">
        <button 
          @click="analysisResults.show = false"
          class="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded"
        >
          Zamknij
        </button>
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
const analysisPolling = ref(null)  // interval ID for polling analysis status
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
  },
  currentLabel: null,  // 'ok' | 'nok' | 'skip' | null
  currentDefectType: null,  // typ wady dla NOK
  autoAdvance: true    // Auto-przejdź po etykietowaniu
})
const labelingStats = ref(null)  // Statystyki etykietowania
const showDefectSelector = ref(false)  // Pokazuje popup wyboru typu wady

// Dostępne typy wad
const defectTypes = [
  { value: 'porosity', label: 'Porowatość', icon: '🫧' },
  { value: 'crack', label: 'Pęknięcie', icon: '💔' },
  { value: 'lack_of_fusion', label: 'Brak przetopu', icon: '🔗' },
  { value: 'undercut', label: 'Podtopienie', icon: '📉' },
  { value: 'burn_through', label: 'Przepalenie', icon: '🔥' },
  { value: 'spatter', label: 'Rozpryski', icon: '💦' },
  { value: 'irregular_bead', label: 'Nierówna spoina', icon: '〰️' },
  { value: 'contamination', label: 'Zanieczyszczenie', icon: '🦠' },
  { value: 'other', label: 'Inna wada', icon: '❓' }
]

// ML state
const mlInfo = ref(null)
const mlPrediction = ref(null)
const mlPredicting = ref(false)
const trainingInProgress = ref(false)
const defectTrainingInProgress = ref(false)
const showingGradCAM = ref(false)

// Defect classifier state
const defectInfo = ref(null)
const defectPrediction = ref(null)
const defectPredicting = ref(false)

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

// ===== VIDEO ANALYSIS FUNCTIONS =====

async function startVideoAnalysis(filename) {
  try {
    // Start analysis
    const response = await fetch(`${API}/ml/analyze-video/${filename}`, { 
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ skip_frames: 5 })  // Analyze every 5th frame for speed
    })
    
    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Nie można rozpocząć analizy')
    }
    
    showToast('🔬 Analiza wideo rozpoczęta')
    
    // Update recording with analysis status - ensure reactivity
    const recording = recordings.value.find(r => r.filename === filename)
    if (recording) {
      // Force reactivity by creating new object
      recording.analysis = { 
        in_progress: true, 
        progress: 0,
        results: null,
        error: null
      }
    }
    
    // Start polling for status
    startAnalysisPolling()
  } catch (e) {
    showToast('❌ ' + e.message, 'error')
  }
}

function startAnalysisPolling() {
  if (analysisPolling.value) return  // Already polling
  
  analysisPolling.value = setInterval(async () => {
    const analyzingRecordings = recordings.value.filter(r => r.analysis?.in_progress)
    
    if (analyzingRecordings.length === 0) {
      stopAnalysisPolling()
      return
    }
    
    for (const rec of analyzingRecordings) {
      try {
        const response = await fetch(`${API}/ml/analyze-video/${rec.filename}/status`)
        if (response.ok) {
          const status = await response.json()
          
          if (status.status === 'completed') {
            // Fetch full results
            const resultsResponse = await fetch(`${API}/ml/analyze-video/${rec.filename}/results`)
            if (resultsResponse.ok) {
              const results = await resultsResponse.json()
              rec.analysis = { in_progress: false, results }
              showToast(`✅ Analiza "${rec.filename}" zakończona`)
            }
          } else if (status.status === 'in_progress') {
            // Update progress - ensure reactivity
            if (!rec.analysis) rec.analysis = {}
            rec.analysis.in_progress = true
            rec.analysis.progress = status.progress || 0
          } else if (status.status === 'error') {
            rec.analysis = { error: status.error || 'Unknown error' }
            showToast(`❌ Błąd analizy "${rec.filename}"`, 'error')
          }
        }
      } catch (e) {
        console.error('Error polling analysis status:', e)
      }
    }
  }, 2000)  // Poll every 2 seconds
}

function stopAnalysisPolling() {
  if (analysisPolling.value) {
    clearInterval(analysisPolling.value)
    analysisPolling.value = null
  }
}

function getAnalysisSummary(recording) {
  if (!recording.analysis?.results) return 'Brak danych'
  
  const { summary, defect_summary } = recording.analysis.results
  let text = `OK: ${summary.ok}, NOK: ${summary.nok}`
  
  if (defect_summary && Object.keys(defect_summary).length > 0) {
    const defects = Object.entries(defect_summary)
      .map(([type, count]) => {
        const defectInfo = defectTypes.find(d => d.value === type)
        return `${defectInfo?.icon || '❓'} ${defectInfo?.label || type}: ${count}`
      })
      .join(', ')
    text += `\nWady: ${defects}`
  }
  
  return text
}

function viewAnalysisResults(filename) {
  const recording = recordings.value.find(r => r.filename === filename)
  if (!recording?.analysis?.results) {
    showToast('❌ Brak wyników analizy', 'error')
    return
  }
  
  // Show detailed results modal
  analysisResults.value = {
    show: true,
    filename,
    results: recording.analysis.results
  }
}

function openFrameInViewer(filename, frameNumber) {
  // Close analysis modal
  analysisResults.value.show = false
  
  // Open frame viewer at specific frame
  openFrameViewer(filename, frameNumber)
}

// Analysis Results Modal state
const analysisResults = ref({
  show: false,
  filename: '',
  results: null
})


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

async function openFrameViewer(filename, startFrame = 0) {
  frameViewer.value.filename = filename
  frameViewer.value.currentFrame = startFrame
  frameViewer.value.loading = true
  frameViewer.value.show = true
  frameViewer.value.currentLabel = null
  mlPrediction.value = null  // Reset ML prediction
  defectPrediction.value = null  // Reset defect prediction
  showingGradCAM.value = false
  resetFilters()
  
  try {
    const response = await fetch(`${API}/recording/${filename}/info`)
    if (!response.ok) throw new Error('Nie można pobrać info o wideo')
    const info = await response.json()
    frameViewer.value.totalFrames = info.frame_count
    updateFrameImage()
    
    // Pobierz statystyki etykietowania, etykietę bieżącej klatki i info ML + defect
    await Promise.all([
      fetchLabelingStats(),
      fetchCurrentLabel(),
      fetchMLInfo(),
      fetchDefectInfo()
    ])
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
    mlPrediction.value = null  // Reset prediction on frame change
    defectPrediction.value = null  // Reset defect prediction
    showingGradCAM.value = false
    updateFrameImage()
    fetchCurrentLabel()
  }
}

function nextFrame() {
  if (frameViewer.value.currentFrame < frameViewer.value.totalFrames - 1) {
    frameViewer.value.currentFrame++
    mlPrediction.value = null  // Reset prediction on frame change
    defectPrediction.value = null  // Reset defect prediction
    showingGradCAM.value = false
    updateFrameImage()
    fetchCurrentLabel()
  }
}

function downloadCurrentFrame() {
  const a = document.createElement('a')
  a.href = frameViewer.value.imageUrl
  a.download = `${frameViewer.value.filename}_frame${frameViewer.value.currentFrame}.jpg`
  a.click()
}

// ============== LABELING ==============

async function labelFrame(label) {
  console.log('labelFrame called with:', label)
  const filename = frameViewer.value.filename
  const frameIndex = frameViewer.value.currentFrame
  console.log('filename:', filename, 'frameIndex:', frameIndex)
  
  try {
    const response = await fetch(`${API}/labeling/${filename}/frame/${frameIndex}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        label: label,
        save_image: true  // Zapisz obraz do folderu treningowego
      })
    })
    
    if (!response.ok) throw new Error('Błąd zapisywania etykiety')
    
    frameViewer.value.currentLabel = label
    frameViewer.value.currentDefectType = null
    await fetchLabelingStats()
    
    // Auto-przejdź do następnej klatki
    if (frameViewer.value.autoAdvance && frameViewer.value.currentFrame < frameViewer.value.totalFrames - 1) {
      frameViewer.value.currentFrame++
      updateFrameImage()
      await fetchCurrentLabel()
    }
    
    const icons = { ok: '✅', nok: '❌', skip: '⏭️' }
    showToast(`${icons[label]} Klatka ${frameIndex} → ${label.toUpperCase()}`)
  } catch (e) {
    showToast('❌ ' + e.message, 'error')
  }
}

// Labelowanie NOK z typem wady
async function labelFrameWithDefect(defectType) {
  console.log('labelFrameWithDefect called with:', defectType)
  const filename = frameViewer.value.filename
  const frameIndex = frameViewer.value.currentFrame
  
  try {
    const response = await fetch(`${API}/labeling/${filename}/frame/${frameIndex}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        label: 'nok',
        defect_type: defectType,
        save_image: true
      })
    })
    
    if (!response.ok) throw new Error('Błąd zapisywania etykiety')
    
    frameViewer.value.currentLabel = 'nok'
    frameViewer.value.currentDefectType = defectType
    showDefectSelector.value = false
    await fetchLabelingStats()
    
    // Auto-przejdź do następnej klatki
    if (frameViewer.value.autoAdvance && frameViewer.value.currentFrame < frameViewer.value.totalFrames - 1) {
      frameViewer.value.currentFrame++
      updateFrameImage()
      await fetchCurrentLabel()
    }
    
    const defect = defectTypes.find(d => d.value === defectType)
    showToast(`❌ NOK - ${defect?.icon} ${defect?.label}`)
  } catch (e) {
    showToast('❌ ' + e.message, 'error')
  }
}

// Pomocnik do wyświetlania nazwy wady
function getDefectLabel(defectType) {
  const defect = defectTypes.find(d => d.value === defectType)
  return defect ? `${defect.icon} ${defect.label}` : defectType
}

async function fetchLabelingStats() {
  try {
    const response = await fetch(`${API}/labeling/stats`)
    if (response.ok) {
      labelingStats.value = await response.json()
    }
  } catch (e) {
    console.error('Failed to fetch labeling stats:', e)
  }
}

async function fetchCurrentLabel() {
  const filename = frameViewer.value.filename
  const frameIndex = frameViewer.value.currentFrame
  
  try {
    const response = await fetch(`${API}/labeling/${filename}/frame/${frameIndex}`)
    if (response.ok) {
      const data = await response.json()
      frameViewer.value.currentLabel = data.label
      frameViewer.value.currentDefectType = data.defect_type || null
    } else {
      frameViewer.value.currentLabel = null
      frameViewer.value.currentDefectType = null
    }
  } catch (e) {
    frameViewer.value.currentLabel = null
    frameViewer.value.currentDefectType = null
  }
}

async function removeLabel() {
  const filename = frameViewer.value.filename
  const frameIndex = frameViewer.value.currentFrame
  
  try {
    const response = await fetch(`${API}/labeling/${filename}/frame/${frameIndex}`, {
      method: 'DELETE'
    })
    
    if (!response.ok) throw new Error('Błąd usuwania etykiety')
    
    frameViewer.value.currentLabel = null
    await fetchLabelingStats()
    showToast('🗑️ Etykieta usunięta')
  } catch (e) {
    showToast('❌ ' + e.message, 'error')
  }
}

// ============== ML CLASSIFICATION ==============

async function fetchMLInfo() {
  try {
    const response = await fetch(`${API}/ml/info`)
    if (response.ok) {
      mlInfo.value = await response.json()
    }
  } catch (e) {
    console.error('Failed to fetch ML info:', e)
  }
}

async function predictFrame() {
  const filename = frameViewer.value.filename
  const frameIndex = frameViewer.value.currentFrame
  
  mlPredicting.value = true
  mlPrediction.value = null
  
  try {
    const response = await fetch(`${API}/ml/predict/${filename}/frame/${frameIndex}?with_gradcam=false`, {
      method: 'POST'
    })
    
    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Błąd predykcji')
    }
    
    mlPrediction.value = await response.json()
    
    const icon = mlPrediction.value.prediction === 'ok' ? '✅' : '❌'
    showToast(`${icon} ${mlPrediction.value.prediction.toUpperCase()}: ${mlPrediction.value.confidence}%`)
  } catch (e) {
    showToast('❌ ' + e.message, 'error')
  } finally {
    mlPredicting.value = false
  }
}

async function showGradCAM() {
  const filename = frameViewer.value.filename
  const frameIndex = frameViewer.value.currentFrame
  
  showingGradCAM.value = true
  
  // Zamień URL obrazu na Grad-CAM overlay
  frameViewer.value.imageUrl = `${API}/ml/predict/${filename}/frame/${frameIndex}/gradcam?alpha=0.5&_t=${Date.now()}`
  
  showToast('🔥 Pokazuję Grad-CAM - obszary uwagi AI')
}

// ============== DEFECT CLASSIFICATION ==============

async function fetchDefectInfo() {
  try {
    const response = await fetch(`${API}/defects/info`)
    if (response.ok) {
      defectInfo.value = await response.json()
    }
  } catch (e) {
    console.error('Failed to fetch defect info:', e)
  }
}

async function predictDefect() {
  const filename = frameViewer.value.filename
  const frameIndex = frameViewer.value.currentFrame
  
  defectPredicting.value = true
  defectPrediction.value = null
  
  try {
    const response = await fetch(`${API}/defects/predict?filename=${filename}&frame_index=${frameIndex}`, {
      method: 'POST'
    })
    
    if (!response.ok) throw new Error('Błąd predykcji defektu')
    
    defectPrediction.value = await response.json()
    showToast(`🔍 ${getDefectLabel(defectPrediction.value.prediction)} (${defectPrediction.value.confidence}%)`)
  } catch (e) {
    showToast('❌ ' + e.message, 'error')
  } finally {
    defectPredicting.value = false
  }
}

async function showDefectGradCAM() {
  const filename = frameViewer.value.filename
  const frameIndex = frameViewer.value.currentFrame
  
  // Zamień URL obrazu na Grad-CAM overlay dla defektów
  frameViewer.value.imageUrl = `${API}/defects/predict/${filename}/frame/${frameIndex}/gradcam?_t=${Date.now()}`
  
  showToast('🔥 Grad-CAM - obszary uwagi dla typu wady')
}

function getTopDefectProbabilities(probabilities, top = 3) {
  return Object.entries(probabilities)
    .sort((a, b) => b[1] - a[1])
    .slice(0, top)
    .reduce((obj, [key, val]) => ({ ...obj, [key]: val }), {})
}

// ============== TRAINING ==============

async function startTraining() {
  trainingInProgress.value = true
  
  try {
    const response = await fetch(`${API}/ml/train?epochs=20&batch_size=16`, {
      method: 'POST'
    })
    
    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Błąd rozpoczęcia treningu')
    }
    
    showToast('🚀 Trening rozpoczęty w tle!')
    
    // Poll status treningu
    const pollTraining = setInterval(async () => {
      const statusResponse = await fetch(`${API}/ml/training-status`)
      if (statusResponse.ok) {
        const status = await statusResponse.json()
        
        if (!status.in_progress) {
          clearInterval(pollTraining)
          trainingInProgress.value = false
          
          if (status.error) {
            showToast('❌ Trening nieudany: ' + status.error, 'error')
          } else {
            showToast(`🎉 Trening zakończony! Dokładność: ${status.history?.best_val_acc?.toFixed(1)}%`)
            await fetchMLInfo()
          }
        }
      }
    }, 3000)
    
  } catch (e) {
    showToast('❌ ' + e.message, 'error')
    trainingInProgress.value = false
  }
}

async function startDefectTraining() {
  defectTrainingInProgress.value = true
  
  try {
    const response = await fetch(`${API}/defects/train?epochs=30&batch_size=16`, {
      method: 'POST'
    })
    
    if (!response.ok) {
      const error = await response.json()
      console.error('Defect training error:', error)
      throw new Error(error.detail || 'Błąd rozpoczęcia treningu klasyfikatora wad')
    }
    
    showToast('🔥 Trening klasyfikatora wad rozpoczęty!')
    
    // Poll status treningu
    const pollDefectTraining = setInterval(async () => {
      const statusResponse = await fetch(`${API}/defects/info`)
      if (statusResponse.ok) {
        const info = await statusResponse.json()
        const status = info.training_status
        
        if (!status.in_progress) {
          clearInterval(pollDefectTraining)
          defectTrainingInProgress.value = false
          
          if (status.error) {
            showToast('❌ Trening wad nieudany: ' + status.error, 'error')
          } else {
            showToast(`🎉 Klasyfikator wad wytrenowany! Dokładność: ${status.history?.best_val_acc?.toFixed(1)}%`)
          }
        }
      }
    }, 3000)
    
  } catch (e) {
    showToast('❌ ' + e.message, 'error')
    defectTrainingInProgress.value = false
  }
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
  stopAnalysisPolling()
})
</script>

<style scoped>
/* Tailwind handles everything */
</style>
