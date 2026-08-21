import { useState, useRef, useCallback } from 'react'
import {
  Mic, Square, Loader2, Send, Keyboard, Layers,
  CheckCircle2, AlertTriangle, Shield, Clock, Zap,
  Activity, Hash, CircleDot, Grid3X3, SplitSquareHorizontal,
  BookOpen, Tags, AlertCircle, ChevronRight
} from 'lucide-react'
import './App.css'

const STRATEGIES = [
  { id: 'fixed', label: 'Fixed', icon: Grid3X3 },
  { id: 'semantic', label: 'Semantic', icon: SplitSquareHorizontal },
  { id: 'window', label: 'Window', icon: BookOpen },
  { id: 'metadata', label: 'Metadata', icon: Tags },
]

const API_BASE = ''

function App() {
  const [strategy, setStrategy] = useState('fixed')
  const [isRecording, setIsRecording] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [textQuery, setTextQuery] = useState('')

  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])

  const startRecording = useCallback(async () => {
    try {
      setError(null)
      setResult(null)

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        }
      })

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
          ? 'audio/webm;codecs=opus'
          : 'audio/webm'
      })

      chunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(track => track.stop())
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' })
        await sendAudio(audioBlob)
      }

      mediaRecorderRef.current = mediaRecorder
      mediaRecorder.start()
      setIsRecording(true)
    } catch (err) {
      setError(`Microphone access denied: ${err.message}`)
    }
  }, [strategy])

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }, [isRecording])

  const toggleRecording = useCallback(() => {
    isRecording ? stopRecording() : startRecording()
  }, [isRecording, startRecording, stopRecording])

  const sendAudio = async (audioBlob) => {
    setIsProcessing(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('audio', audioBlob, 'recording.webm')
      formData.append('strategy', strategy)

      const response = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        throw new Error(errData.detail || `Server error: ${response.status}`)
      }

      setResult(await response.json())
    } catch (err) {
      setError(`Request failed: ${err.message}`)
    } finally {
      setIsProcessing(false)
    }
  }

  const sendTextQuery = async (e) => {
    e.preventDefault()
    if (!textQuery.trim() || isProcessing) return

    setIsProcessing(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('query', textQuery.trim())
      formData.append('strategy', strategy)

      const response = await fetch(`${API_BASE}/ask-text`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        throw new Error(errData.detail || `Server error: ${response.status}`)
      }

      setResult(await response.json())
      setTextQuery('')
    } catch (err) {
      setError(`Request failed: ${err.message}`)
    } finally {
      setIsProcessing(false)
    }
  }

  const getLatencyClass = (ms) => {
    if (ms < 200) return ''
    if (ms < 500) return 'slow'
    return 'very-slow'
  }

  const getGroundednessClass = (score) => {
    if (score >= 0.4) return 'high'
    if (score >= 0.2) return 'medium'
    return 'low'
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-icon">
          <Activity size={22} />
        </div>
        <h1>Voice <span>RAG</span></h1>
        <p>Speak a question, get a grounded answer</p>
        <span className="badge">
          <CircleDot size={12} />
          HH Goa 2026 — Task 2
        </span>
      </header>

      {/* Main Card */}
      <div className="main-card">
        {/* Strategy Selector */}
        <div className="strategy-selector">
          <label>
            <Layers size={13} />
            Chunking Strategy
          </label>
          <div className="strategy-options">
            {STRATEGIES.map(s => {
              const Icon = s.icon
              return (
                <button
                  key={s.id}
                  className={`strategy-btn ${strategy === s.id ? 'active' : ''}`}
                  onClick={() => setStrategy(s.id)}
                  disabled={isProcessing}
                >
                  <Icon size={16} />
                  {s.label}
                </button>
              )
            })}
          </div>
        </div>

        {/* Mic Button */}
        <div className="mic-container">
          <button
            className={`mic-button ${isRecording ? 'recording' : ''}`}
            onClick={toggleRecording}
            disabled={isProcessing}
            aria-label={isRecording ? 'Stop recording' : 'Start recording'}
          >
            {isRecording ? <Square size={24} /> : isProcessing ? <Loader2 size={28} className="spin" /> : <Mic size={28} />}
          </button>
          <span className={`mic-status ${isRecording ? 'recording' : ''} ${isProcessing ? 'processing' : ''}`}>
            {isRecording ? (
              <><CircleDot size={14} /> Recording — tap to stop</>
            ) : isProcessing ? (
              <><Loader2 size={14} /> Processing your question...</>
            ) : (
              <><Mic size={14} /> Tap to speak your question</>
            )}
          </span>
        </div>

        {/* Text Input */}
        <details className="text-input-section">
          <summary>
            <Keyboard size={14} />
            Or type your question instead
          </summary>
          <form className="text-form" onSubmit={sendTextQuery}>
            <input
              type="text"
              className="text-input"
              value={textQuery}
              onChange={(e) => setTextQuery(e.target.value)}
              placeholder="Type your question in Hindi..."
              disabled={isProcessing}
            />
            <button
              type="submit"
              className="text-submit"
              disabled={isProcessing || !textQuery.trim()}
            >
              <Send size={15} />
              Ask
            </button>
          </form>
        </details>
      </div>

      {/* Error */}
      {error && (
        <div className="error-card">
          <AlertCircle size={18} />
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="results">
          <div className={`result-card ${result.abstained ? 'abstained' : ''}`}>
            {/* Transcript */}
            {result.transcript && (
              <>
                <div className="result-label">
                  <Mic size={13} />
                  Transcript
                </div>
                <div className="result-transcript">"{result.transcript}"</div>
              </>
            )}

            {/* Answer */}
            <div className={`result-label ${result.abstained ? 'warning' : 'success'}`}>
              {result.abstained
                ? <><AlertTriangle size={13} /> Abstained</>
                : <><CheckCircle2 size={13} /> Answer</>
              }
            </div>
            <div className={`result-answer ${result.abstained ? 'abstain-message' : ''}`}>
              {result.answer}
            </div>

            {/* Groundedness */}
            {!result.abstained && result.groundedness_score > 0 && (
              <div className={`groundedness ${getGroundednessClass(result.groundedness_score)}`}>
                <Shield size={12} />
                Groundedness: {(result.groundedness_score * 100).toFixed(0)}%
              </div>
            )}

            {/* Latency Stats */}
            <div className="latency-grid">
              {result.stage_latencies_ms?.retrieval !== undefined && (
                <div className="latency-item">
                  <div className={`latency-value ${getLatencyClass(result.stage_latencies_ms.retrieval)}`}>
                    {result.stage_latencies_ms.retrieval.toFixed(0)}ms
                  </div>
                  <div className="latency-label">Retrieval</div>
                </div>
              )}
              {result.stage_latencies_ms?.stt !== undefined && (
                <div className="latency-item">
                  <div className={`latency-value ${getLatencyClass(result.stage_latencies_ms.stt)}`}>
                    {result.stage_latencies_ms.stt.toFixed(0)}ms
                  </div>
                  <div className="latency-label">STT</div>
                </div>
              )}
              {result.stage_latencies_ms?.generation !== undefined && (
                <div className="latency-item">
                  <div className={`latency-value ${getLatencyClass(result.stage_latencies_ms.generation)}`}>
                    {result.stage_latencies_ms.generation.toFixed(0)}ms
                  </div>
                  <div className="latency-label">Generation</div>
                </div>
              )}
              {result.total_latency_ms !== undefined && (
                <div className="latency-item">
                  <div className={`latency-value ${getLatencyClass(result.total_latency_ms)}`}>
                    {result.total_latency_ms.toFixed(0)}ms
                  </div>
                  <div className="latency-label">Total</div>
                </div>
              )}
            </div>

            {/* Strategy Info */}
            <div className="strategy-info">
              <Layers size={12} />
              Strategy: <strong>{result.strategy_used}</strong>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="footer">
        <p>Built for HH Goa 2026 · <span className="hashtag">#RAGInGoa</span></p>
        <p style={{ marginTop: '0.25rem' }}>Voice-Enabled RAG with 4 Chunking Strategies</p>
      </footer>
    </div>
  )
}

export default App
