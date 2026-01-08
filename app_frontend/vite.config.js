import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    host: '0.0.0.0',
    port: 3000,
    proxy: {
      '/camera': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/recording': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/edge': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/labeling': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/ml': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
})
