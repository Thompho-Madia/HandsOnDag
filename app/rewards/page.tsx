export default function RewardsPage(){
  return (
    <div>
      <div className="card">
        <h2>Rewards & Points</h2>
        <p className="small">Earn points by submitting predictions. Points unlock advanced features like detailed charts and accuracy tracking.</p>
      </div>

      <div className="card">
        <h3>Your points (sample)</h3>
        <p style={{fontSize:20,fontWeight:700}}>150 pts</p>
        <ul className="small">
          <li>+50 pts – Submit your first prediction</li>
          <li>+100 pts – Store 5 predictions</li>
          <li>+200 pts – Correct prediction</li>
        </ul>
      </div>

      <div className="card small">
        <strong>Unlocks</strong>
        <p>At 200 pts you'll get access to accuracy tracking and detailed charts (not implemented in this sample).</p>
      </div>
    </div>
  )
}
