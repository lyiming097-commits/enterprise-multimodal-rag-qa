import { createApp } from 'vue'
import App from './App.vue'
import './styles.css'

const app = createApp(App)
app.config.errorHandler = (error) => {
  console.error('RAG Studio 渲染错误', error)
  const root = document.querySelector('#app')
  if (root && !root.textContent?.trim()) {
    root.innerHTML = '<div class="fatal-error">页面渲染遇到问题，请刷新后重试。</div>'
  }
}
app.mount('#app')
