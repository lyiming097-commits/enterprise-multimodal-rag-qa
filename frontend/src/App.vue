<script setup>
import { computed, nextTick, onMounted, ref, watch } from 'vue'

const health = ref(null)
const knowledgeBases = ref([])
const selectedId = ref('')
const documents = ref([])
const messages = ref([])
const lastDebug = ref(null)
const newKnowledgeBase = ref('')
const question = ref('')
const busy = ref(false)
const streamStage = ref('正在执行六步检索…')
const notice = ref(null)
const filesInput = ref(null)
const chatScroll = ref(null)
const mobilePanel = ref('chat')
const chunkModal = ref(false)
const chunkDocument = ref(null)
const chunks = ref([])
const chunkTotal = ref(0)
const chunkLoading = ref(false)

const selectedKnowledgeBase = computed(() =>
  knowledgeBases.value.find((item) => item.id === selectedId.value),
)
const hasChunks = computed(() => documents.value.some((item) => item.chunk_count > 0))

const canAsk = computed(
  () => selectedId.value && question.value.trim() && !busy.value && hasChunks.value,
)

const pipelineSteps = computed(() => {
  const debug = lastDebug.value
  const trace = debug?.retrieval_trace || []
  const reranked = trace.filter((item) => item.rerank_score !== null)
  return [
    {
      number: '01',
      title: 'Query 预处理',
      subtitle: '改写 · HyDE · 多角度扩写',
      ready: Boolean(debug?.query_plan),
    },
    {
      number: '02',
      title: 'Query Embedding',
      subtitle: health.value?.embedding
        ? `${health.value.embedding.model} · ${health.value.embedding.dimension}D`
        : 'Qwen3 Embedding',
      ready: Boolean(debug),
    },
    {
      number: '03',
      title: '多路召回 + RRF',
      subtitle: debug ? `${trace.length} 个候选结果` : 'pgvector · BM25',
      ready: trace.length > 0,
    },
    {
      number: '04',
      title: 'Cross-Encoder 精排',
      subtitle: debug ? `${reranked.length} 个候选已打分` : health.value?.reranker?.message,
      ready: reranked.length > 0,
    },
    {
      number: '05',
      title: '受约束 Prompt',
      subtitle: debug ? `路由：${routeLabel(debug.route)}` : '仅依据检索证据',
      ready: Boolean(debug),
    },
    {
      number: '06',
      title: '生成与溯源',
      subtitle: debug ? '已完成回答生成' : 'DeepSeek · 基于证据生成',
      ready: Boolean(debug),
    },
  ]
})

onMounted(async () => {
  await Promise.all([loadHealth(), loadKnowledgeBases()])
})

watch(selectedId, async (next, previous) => {
  if (next === previous) return
  messages.value = loadConversation(next)
  lastDebug.value = null
  await loadDocuments()
})

watch(
  messages,
  (value) => {
    if (!selectedId.value) return
    const saved = value
      .filter((item) => item.role === 'user' || item.role === 'assistant')
      .map(({ role, content, citations, meta }) => ({ role, content, citations, meta }))
    localStorage.setItem(`rag-studio:conversation:${selectedId.value}`, JSON.stringify(saved))
  },
  { deep: true },
)

function loadConversation(knowledgeBaseId) {
  if (!knowledgeBaseId) return []
  try {
    const saved = JSON.parse(localStorage.getItem(`rag-studio:conversation:${knowledgeBaseId}`) || '[]')
    return Array.isArray(saved)
      ? saved.map((item) => ({ ...item, streaming: false })).filter((item) => item.content)
      : []
  } catch {
    return []
  }
}

function clearConversation() {
  messages.value = []
  lastDebug.value = null
  if (selectedId.value) localStorage.removeItem(`rag-studio:conversation:${selectedId.value}`)
}

async function api(path, options = {}) {
  const response = await fetch(path, options)
  const contentType = response.headers.get('content-type') || ''
  const payload = contentType.includes('application/json') ? await response.json() : null
  if (!response.ok) {
    throw new Error(payload?.detail || `请求失败 (${response.status})`)
  }
  return payload
}

async function loadHealth() {
  try {
    health.value = await api('/api/health')
  } catch (error) {
    showNotice(error.message, 'error')
  }
}

async function loadKnowledgeBases(preferredId = '') {
  try {
    knowledgeBases.value = await api('/api/knowledge-bases')
    const target = preferredId || selectedId.value
    if (target && knowledgeBases.value.some((item) => item.id === target)) {
      selectedId.value = target
    } else {
      selectedId.value = knowledgeBases.value[0]?.id || ''
    }
    if (selectedId.value) await loadDocuments()
  } catch (error) {
    showNotice(error.message, 'error')
  }
}

async function loadDocuments() {
  if (!selectedId.value) {
    documents.value = []
    return
  }
  try {
    documents.value = await api(`/api/knowledge-bases/${selectedId.value}/documents`)
  } catch (error) {
    showNotice(error.message, 'error')
  }
}

async function createKnowledgeBase() {
  const name = newKnowledgeBase.value.trim()
  if (!name) return
  busy.value = true
  try {
    const created = await api('/api/knowledge-bases', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    })
    newKnowledgeBase.value = ''
    await loadKnowledgeBases(created.id)
    showNotice(`已创建知识库「${created.name}」`)
  } catch (error) {
    showNotice(error.message, 'error')
  } finally {
    busy.value = false
  }
}

async function uploadFiles(event) {
  const files = Array.from(event.target.files || [])
  if (!files.length || !selectedId.value) return
  const form = new FormData()
  files.forEach((file) => form.append('files', file))
  busy.value = true
  showNotice(`正在解析并索引 ${files.length} 个文件…`, 'working')
  try {
    const result = await api(`/api/knowledge-bases/${selectedId.value}/documents`, {
      method: 'POST',
      body: form,
    })
    const succeeded = result.results.filter((item) => item.ok).length
    const failed = result.results.filter((item) => !item.ok)
    await Promise.all([loadDocuments(), loadKnowledgeBases(selectedId.value)])
    if (failed.length) {
      showNotice(`成功 ${succeeded} 个，失败 ${failed.length} 个：${failed[0].message}`, 'error')
    } else {
      showNotice(`已完成 ${succeeded} 个文件的索引`)
    }
  } catch (error) {
    showNotice(error.message, 'error')
  } finally {
    busy.value = false
    if (filesInput.value) filesInput.value.value = ''
  }
}

async function loadSample() {
  if (!selectedId.value) return
  busy.value = true
  showNotice('正在向量化示例资料…', 'working')
  try {
    const result = await api(`/api/knowledge-bases/${selectedId.value}/sample`, { method: 'POST' })
    await Promise.all([loadDocuments(), loadKnowledgeBases(selectedId.value)])
    showNotice(result.message)
  } catch (error) {
    showNotice(error.message, 'error')
  } finally {
    busy.value = false
  }
}

async function removeDocument(document) {
  if (!window.confirm(`确定从知识库移除「${document.filename}」吗？`)) return
  busy.value = true
  try {
    await api(`/api/documents/${document.id}`, { method: 'DELETE' })
    await Promise.all([loadDocuments(), loadKnowledgeBases(selectedId.value)])
    showNotice('文档已移除')
  } catch (error) {
    showNotice(error.message, 'error')
  } finally {
    busy.value = false
  }
}

async function openChunkViewer(document) {
  chunkDocument.value = document
  chunkModal.value = true
  await loadChunks(document)
}

function closeChunkViewer() {
  chunkModal.value = false
  chunkDocument.value = null
  chunks.value = []
}

async function loadChunks(document = chunkDocument.value) {
  if (!document || !selectedId.value) return
  chunkLoading.value = true
  try {
    const result = await api(
      `/api/knowledge-bases/${selectedId.value}/chunks?document_id=${document.id}&include_inactive=true&limit=500`,
    )
    chunks.value = result.items
    chunkTotal.value = result.total
  } catch (error) {
    showNotice(error.message, 'error')
  } finally {
    chunkLoading.value = false
  }
}

async function removeChunk(chunk) {
  if (!chunk.is_active || !window.confirm(`确定删除第 ${chunk.ordinal} 个分片吗？`)) return
  try {
    await api(`/api/chunks/${chunk.id}`, { method: 'DELETE' })
    await Promise.all([loadChunks(), loadDocuments(), loadKnowledgeBases(selectedId.value)])
    showNotice('分片已删除，检索将自动忽略它')
  } catch (error) {
    showNotice(error.message, 'error')
  }
}

async function askQuestion(preset = '') {
  const content = (preset || question.value).trim()
  if (!content || busy.value || !selectedId.value) return
  question.value = ''
  messages.value.push({ role: 'user', content })
  const history = messages.value.slice(0, -1).map(({ role, content: text }) => ({
    role,
    content: text,
  }))
  const assistant = { role: 'assistant', content: '', citations: [], streaming: true }
  messages.value.push(assistant)
  const assistantIndex = messages.value.length - 1
  const updateAssistant = (update) => {
    const current = messages.value[assistantIndex]
    if (current?.role === 'assistant') update(current)
  }
  busy.value = true
  streamStage.value = '正在执行六步检索…'
  await scrollToBottom()
  try {
    const response = await fetch(`/api/knowledge-bases/${selectedId.value}/ask/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: content, history }),
    })
    if (!response.ok) {
      const payload = await response.json().catch(() => null)
      throw new Error(payload?.detail || `请求失败 (${response.status})`)
    }
    const reader = response.body?.getReader()
    if (!reader) throw new Error('浏览器不支持流式响应')
    const decoder = new TextDecoder()
    let buffer = ''
    const consume = (raw) => {
      buffer += raw
      const blocks = buffer.split(/\r?\n\r?\n/)
      buffer = blocks.pop() || ''
      blocks.forEach((block) => {
        const data = block
          .split(/\r?\n/)
          .filter((line) => line.startsWith('data:'))
          .map((line) => line.slice(5).trim())
          .join('\n')
        if (!data) return
        const event = JSON.parse(data)
        if (event.type === 'stage') streamStage.value = event.message
        if (event.type === 'query_plan') lastDebug.value = { ...(lastDebug.value || {}), query_plan: event.query_plan }
        if (event.type === 'retrieval') lastDebug.value = { ...(lastDebug.value || {}), retrieval_trace: event.retrieval_trace }
        if (event.type === 'token') {
          updateAssistant((current) => {
            current.content += event.content || ''
          })
          void scrollToBottom()
        }
        if (event.type === 'done') {
          const result = event.result
          updateAssistant((current) => {
            current.content = result.answer
            current.citations = result.citations
            current.meta = { route: result.route, count: result.retrieved_count, secondRetrieval: result.second_retrieval }
            current.streaming = false
          })
          lastDebug.value = result
        }
        if (event.type === 'error') throw new Error(event.message || '流式问答失败')
      })
    }
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      consume(decoder.decode(value, { stream: true }))
    }
    consume(decoder.decode())
    updateAssistant((current) => {
      current.streaming = false
    })
    mobilePanel.value = 'chat'
  } catch (error) {
    if (messages.value[assistantIndex]?.role === 'assistant') messages.value.splice(assistantIndex, 1)
    messages.value.push({ role: 'error', content: error.message })
  } finally {
    busy.value = false
    streamStage.value = ''
    await scrollToBottom()
  }
}

async function scrollToBottom() {
  await nextTick()
  if (chatScroll.value) chatScroll.value.scrollTop = chatScroll.value.scrollHeight
}

function showNotice(message, type = 'success') {
  notice.value = { message, type }
  if (type !== 'working') {
    window.setTimeout(() => {
      if (notice.value?.message === message) notice.value = null
    }, 4000)
  }
}

function renderAnswer(content) {
  const escaped = escapeHtml(content)
  return escaped
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h2>$1</h2>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\[S\d+\]/g, '')
    .replace(/\n/g, '<br>')
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;')
}

function routeLabel(route) {
  return { lookup: '事实查询', comparison: '对比分析', summary: '总结概括' }[route] || route
}

function score(value) {
  if (value === null || value === undefined) return '—'
  return Number(value).toFixed(4)
}

function documentIcon(filename) {
  const suffix = filename.split('.').pop()?.toLowerCase()
  if (suffix === 'pdf') return 'PDF'
  if (['png', 'jpg', 'jpeg', 'webp'].includes(suffix)) return 'IMG'
  if (['md', 'markdown'].includes(suffix)) return 'MD'
  return 'TXT'
}
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark"><span></span><span></span><span></span></div>
        <div>
          <strong>RAG Studio</strong>
          <small>Six-step retrieval lab</small>
        </div>
      </div>

      <div class="system-status">
        <div class="status-item" :class="{ online: health?.database?.ok }">
          <i></i><span>{{ health?.database?.message || '检查 PostgreSQL…' }}</span>
        </div>
        <div class="status-item" :class="{ online: health?.ollama?.ok }">
          <i></i><span>{{ health?.ollama?.message || '检查 Ollama…' }}</span>
        </div>
        <div class="status-item" :class="{ online: health?.deepseek?.ok }">
          <i></i><span>{{ health?.deepseek?.message || '检查 DeepSeek…' }}</span>
        </div>
        <div class="status-item" :class="{ online: health?.ocr?.ok }">
          <i></i><span>{{ health?.ocr?.message || '检查 OCR…' }}</span>
        </div>
        <div class="status-item" :class="{ online: health?.vlm?.ok }">
          <i></i><span>{{ health?.vlm?.message || '检查 VLM…' }}</span>
        </div>
      </div>
    </header>

    <nav class="mobile-tabs">
      <button :class="{ active: mobilePanel === 'library' }" @click="mobilePanel = 'library'">资料库</button>
      <button :class="{ active: mobilePanel === 'chat' }" @click="mobilePanel = 'chat'">问答</button>
      <button :class="{ active: mobilePanel === 'trace' }" @click="mobilePanel = 'trace'">检索链路</button>
    </nav>

    <main class="workspace">
      <aside class="library-panel" :class="{ 'mobile-active': mobilePanel === 'library' }">
        <section class="panel-heading">
          <div>
            <span class="eyebrow">KNOWLEDGE BASE</span>
            <h2>我的资料库</h2>
          </div>
          <span class="count-badge">{{ knowledgeBases.length }}</span>
        </section>

        <div class="create-row">
          <input
            v-model="newKnowledgeBase"
            placeholder="新知识库名称"
            maxlength="200"
            @keyup.enter="createKnowledgeBase"
          />
          <button :disabled="busy || !newKnowledgeBase.trim()" title="创建" @click="createKnowledgeBase">+</button>
        </div>

        <div v-if="knowledgeBases.length" class="kb-list">
          <button
            v-for="kb in knowledgeBases"
            :key="kb.id"
            class="kb-card"
            :class="{ active: selectedId === kb.id }"
            @click="selectedId = kb.id"
          >
            <span class="kb-icon">◇</span>
            <span class="kb-copy">
              <strong>{{ kb.name }}</strong>
              <small>{{ kb.document_count }} 个文档 · {{ kb.chunk_count }} 个分片</small>
            </span>
            <span class="kb-arrow">›</span>
          </button>
        </div>
        <div v-else class="empty-library">
          <div class="empty-glyph">◇</div>
          <strong>创建第一个知识库</strong>
          <p>然后导入你的资料，开始可溯源问答。</p>
        </div>

        <div v-if="selectedKnowledgeBase" class="document-section">
          <div class="section-title">
            <span>已索引资料</span>
            <small>{{ documents.length }}</small>
          </div>
          <div class="document-list">
            <article v-for="document in documents" :key="document.id" class="document-card">
              <span class="file-type">{{ documentIcon(document.filename) }}</span>
              <span class="document-copy">
                <strong :title="document.filename">{{ document.filename }}</strong>
                <small>{{ document.chunk_count }} chunks · v{{ document.version }}</small>
                <span v-if="document.extraction?.used_ocr" class="method-tag ocr">OCR</span>
                <span v-if="document.extraction?.used_vlm" class="method-tag vlm">VLM</span>
              </span>
              <div class="document-actions">
                <button class="chunk-button" title="查看分片" @click="openChunkViewer(document)">分片</button>
                <button class="icon-button" title="移除" @click="removeDocument(document)">×</button>
              </div>
            </article>
          </div>

          <input
            ref="filesInput"
            class="file-input"
            type="file"
            multiple
            accept=".pdf,.md,.markdown,.txt,.png,.jpg,.jpeg,.webp,.bmp,.tiff"
            @change="uploadFiles"
          />
          <button class="upload-button" :disabled="busy" @click="filesInput?.click()">
            <span>↑</span>上传本地资料
          </button>
          <button class="sample-button" :disabled="busy" @click="loadSample">
            没有资料？一键加载测试样本
          </button>
          <p class="upload-hint">PDF / Markdown / TXT / 图片 · 单文件最大 30 MB</p>
        </div>
      </aside>

      <section class="chat-panel" :class="{ 'mobile-active': mobilePanel === 'chat' }">
        <div class="chat-header">
          <div>
            <span class="eyebrow">CONVERSATION</span>
            <h1>{{ selectedKnowledgeBase?.name || '知识问答' }}</h1>
          </div>
          <button v-if="messages.length" class="text-button" @click="clearConversation">
            清空对话
          </button>
        </div>

        <div ref="chatScroll" class="chat-scroll">
          <div v-if="!selectedKnowledgeBase" class="welcome-state">
            <div class="orb"><span></span></div>
            <span class="eyebrow">WELCOME TO RAG STUDIO</span>
            <h2>先创建一个知识库</h2>
            <p>你的资料会在本地向量化，并通过六步 RAG 链路生成带引用的答案。</p>
          </div>

          <div v-else-if="!hasChunks" class="welcome-state">
            <div class="orb document-orb"><span></span></div>
            <span class="eyebrow">READY TO INDEX</span>
            <h2>导入第一份资料</h2>
            <p>在左侧上传文件，或一键加载项目测试样本。</p>
            <button class="primary-action" :disabled="busy" @click="loadSample">加载测试样本</button>
          </div>

          <div v-else-if="!messages.length" class="welcome-state ready-state">
            <div class="orb ready-orb"><span></span></div>
            <span class="eyebrow">ASK YOUR KNOWLEDGE</span>
            <h2>从你的资料里得到可验证的答案</h2>
            <p>每个关键事实都会映射到真实文档片段。先试试下面的问题。</p>
            <div class="prompt-suggestions">
              <button @click="askQuestion('这个项目使用什么向量数据库和向量化模型？')">
                项目使用什么向量数据库和模型？
              </button>
              <button @click="askQuestion('请概括这个知识库的主要内容')">概括知识库的主要内容</button>
            </div>
          </div>

          <div v-else class="message-list">
            <article v-for="(message, index) in messages" :key="index" class="message" :class="message.role">
              <div class="avatar">{{ message.role === 'user' ? '你' : message.role === 'error' ? '!' : 'R' }}</div>
              <div class="message-body">
                <div class="message-label">
                  {{ message.role === 'user' ? '你' : message.role === 'error' ? '运行异常' : 'RAG Studio' }}
                  <span v-if="message.meta">{{ routeLabel(message.meta.route) }} · {{ message.meta.count }} 个候选</span>
                </div>
                <div v-if="message.role === 'assistant' && message.content" class="answer-content" v-html="renderAnswer(message.content)"></div>
                <div v-else-if="message.role === 'assistant'" class="thinking"><i></i><i></i><i></i><span>{{ streamStage || '正在生成回答…' }}</span></div>
                <div v-else class="plain-content">{{ message.content }}</div>
              </div>
            </article>

          </div>
        </div>

        <form class="composer" @submit.prevent="askQuestion()">
          <div class="composer-box" :class="{ disabled: !hasChunks }">
            <textarea
              v-model="question"
              :disabled="!hasChunks || busy"
              :placeholder="hasChunks ? '输入关于当前知识库的问题…' : '请先导入资料或恢复分片'"
              rows="1"
              @keydown.enter.exact.prevent="askQuestion()"
            ></textarea>
            <button type="submit" :disabled="!canAsk" title="发送">
              <span>↑</span>
            </button>
          </div>
          <p>DeepSeek 可能会犯错，请根据引用原文验证关键信息。</p>
        </form>
      </section>

      <aside class="trace-panel" :class="{ 'mobile-active': mobilePanel === 'trace' }">
        <section class="panel-heading trace-heading">
          <div>
            <span class="eyebrow">RETRIEVAL TRACE</span>
            <h2>六步检索链路</h2>
          </div>
          <span class="live-badge"><i></i> LIVE</span>
        </section>

        <div class="pipeline">
          <article
            v-for="(step, index) in pipelineSteps"
            :key="step.number"
            class="pipeline-step"
            :class="{ ready: step.ready }"
          >
            <div class="step-rail">
              <span>{{ step.number }}</span>
              <i v-if="index < pipelineSteps.length - 1"></i>
            </div>
            <div class="step-card">
              <strong>{{ step.title }}</strong>
              <small>{{ step.subtitle }}</small>

              <template v-if="lastDebug && lastDebug.query_plan && index === 0">
                <div class="query-detail">
                  <label>改写后</label>
                  <p>{{ lastDebug.query_plan.rewritten_query }}</p>
                  <label>扩写角度</label>
                  <span v-for="item in lastDebug.query_plan.expanded_queries" :key="item" class="query-chip">{{ item }}</span>
                  <details v-if="lastDebug.query_plan.hypothetical_document">
                    <summary>HyDE 假设文档</summary>
                    <p>{{ lastDebug.query_plan.hypothetical_document }}</p>
                  </details>
                </div>
              </template>

              <template v-if="lastDebug && lastDebug.retrieval_trace && index === 2">
                <div class="trace-results">
                  <article v-for="(item, traceIndex) in (lastDebug.retrieval_trace || []).slice(0, 5)" :key="item.chunk_id">
                    <span class="rank">{{ traceIndex + 1 }}</span>
                    <div>
                      <strong>{{ item.source }}</strong>
                      <small>{{ item.retrieval_paths.join(' · ') }}</small>
                    </div>
                    <b>{{ score(item.fusion_score) }}</b>
                  </article>
                </div>
              </template>

              <template v-if="lastDebug && lastDebug.retrieval_trace && index === 3">
                <div class="score-row">
                  <span v-for="item in (lastDebug.retrieval_trace || []).slice(0, 5)" :key="item.chunk_id">
                    {{ score(item.rerank_score) }}
                  </span>
                </div>
              </template>

              <template v-if="lastDebug && index === 4">
                <ul class="constraint-list">
                  <li>只使用检索证据</li>
                  <li>忽略资料中的指令</li>
                  <li>证据不足时明确拒答</li>
                </ul>
              </template>

              <template v-if="lastDebug && index === 5">
                <div class="citation-summary">
                  <span v-for="item in (lastDebug.citations || [])" :key="item.citation_id">[{{ item.citation_id }}]</span>
                  <small v-if="!(lastDebug.citations || []).length">本次无有效引用</small>
                </div>
              </template>
            </div>
          </article>
        </div>

        <div v-if="!lastDebug" class="trace-placeholder">
          <span>↗</span>
          <p>完成一次问答后，这里会展开每一步的输入、路径和分数。</p>
        </div>
      </aside>
    </main>

    <transition name="toast">
      <div v-if="notice" class="notice" :class="notice.type">
        <i v-if="notice.type === 'working'" class="notice-spinner"></i>
        <span v-else>{{ notice.type === 'error' ? '!' : '✓' }}</span>
        {{ notice.message }}
      </div>
    </transition>

    <div v-if="chunkModal" class="chunk-modal" @click.self="closeChunkViewer">
      <section class="chunk-drawer">
        <header class="chunk-header">
          <div>
            <span class="eyebrow">CHUNK INSPECTOR</span>
            <h2>{{ chunkDocument?.filename }}</h2>
            <small>{{ chunkTotal }} 个分片 · 已删除的分片仍保留在历史记录中</small>
          </div>
          <button class="text-button" @click="closeChunkViewer">关闭</button>
        </header>
        <div v-if="chunkLoading" class="chunk-empty">正在加载分片…</div>
        <div v-else-if="!chunks.length" class="chunk-empty">暂无分片</div>
        <div v-else class="chunk-list">
          <article v-for="chunk in chunks" :key="chunk.id" class="chunk-card" :class="{ inactive: !chunk.is_active }">
            <div class="chunk-meta">
              <strong>#{{ chunk.ordinal }}</strong>
              <span>v{{ chunk.version }}<template v-if="chunk.page"> · P.{{ chunk.page }}</template></span>
              <span
                v-if="chunk.metadata?.element_type"
                class="method-tag"
                :class="chunk.metadata.element_type.includes('vlm') ? 'vlm' : chunk.metadata.element_type.includes('ocr') ? 'ocr' : ''"
              >{{ chunk.metadata.element_type }}</span>
              <span v-if="!chunk.is_active" class="inactive-tag">已删除</span>
              <button v-if="chunk.is_active" class="chunk-delete" @click="removeChunk(chunk)">删除</button>
            </div>
            <div v-if="chunk.section" class="chunk-section">{{ chunk.section }}</div>
            <p class="chunk-content">{{ chunk.content }}</p>
            <small class="chunk-foot">{{ chunk.token_count }} tokens · {{ chunk.embedding_model || '未记录模型' }}</small>
          </article>
        </div>
      </section>
    </div>
  </div>
</template>
