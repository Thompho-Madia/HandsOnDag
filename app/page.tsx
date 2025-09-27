'use client'

import { useState } from 'react'

export default function HomePage() {
  const [pair, setPair] = useState('USD/EUR')
  const [prediction, setPrediction] = useState<null | { direction: string; confidence: number; risk: string }>(null)
  const [publishing, setPublishing] = useState(false)

  function makePrediction() {
    // simple simulated prediction
    const dir = Math.random() > 0.5 ? 'BUY' : 'SELL'
    const conf = Math.floor(50 + Math.random() * 50)
    const risk = conf > 75 ? 'low' : conf > 55 ? 'medium' : 'high'
    setPrediction({ direction: dir, confidence: conf, risk })
  }

  async function publishPrediction() {
    setPublishing(true)
    // placeholder for on-chain publish flow
    await new Promise((r) => setTimeout(r, 900))
    alert('Prediction published on-chain (simulated)')
    setPublishing(false)
  }

  return (
    <div>
      <div className="card">
        <h2>Predict a currency pair</h2>
        <p className="small">Enter a currency pair and get a quick AI-powered prediction (simulated in this sample).</p>
        <div style={{marginTop:12}} className="row">
          <input className="input" value={pair} onChange={(e) => setPair(e.target.value)} />
          <button className="btn" onClick={makePrediction}>Get Prediction</button>
        </div>
      </div>

      {prediction && (
        <div className="card">
          <h3>Prediction for {pair}</h3>
          <p><span className="prediction-badge">{prediction.direction}</span> &nbsp; Confidence: <strong>{prediction.confidence}%</strong></p>
          <p className="small">Risk level: <strong>{prediction.risk}</strong></p>
          <div style={{marginTop:12}} className="row">
            <button className="btn" onClick={publishPrediction} disabled={publishing}>{publishing ? 'Publishing...' : 'Publish Prediction'}</button>
            <button className="btn" onClick={() => navigator.clipboard?.writeText(JSON.stringify({ pair, ...prediction, timestamp: Date.now() }))}>Copy JSON</button>
          </div>
        </div>
      )}

      <div className="card">
        <h3>Why PrediXchain?</h3>
        <p className="small">This sample implements the public-facing pages from the MVP spec: Home, Prediction Feed, Leaderboard, and Rewards. The publish flow is simulated — integrate a backend and smart contract to store real predictions on-chain.</p>
      </div>
    </div>
  )
}
