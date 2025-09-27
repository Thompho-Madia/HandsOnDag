export default function PredictionsPage(){
  const sample = [
    {id:1, pair:'USD/EUR', dir:'BUY', conf:72, risk:'medium', by:'0x4F...9b2', ts:1695820800},
    {id:2, pair:'GBP/USD', dir:'SELL', conf:61, risk:'medium', by:'0xAb...12e', ts:1695907200},
  ]
  return (
    <div>
      <div className="card">
        <h2>Prediction Feed</h2>
        <p className="small">Recent predictions stored on-chain (sample data). Click any item to inspect on a real chain explorer.</p>
      </div>

      {sample.map(p=> (
        <div key={p.id} className="card" style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
          <div>
            <div style={{fontWeight:700}}>{p.pair} <span style={{marginLeft:8}} className="small">by {p.by}</span></div>
            <div className="small">{p.dir} — Confidence: {p.conf}% — Risk: {p.risk}</div>
          </div>
          <div className="small">{new Date(p.ts*1000).toLocaleString()}</div>
        </div>
      ))}
    </div>
  )
}
