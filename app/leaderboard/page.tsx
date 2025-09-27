export default function LeaderboardPage(){
  const leaders = [
    {rank:1, addr:'0xFf...a1', score:1240},
    {rank:2, addr:'0xAA...b2', score:980},
    {rank:3, addr:'0xCC...c3', score:760},
  ]
  return (
    <div>
      <div className="card"><h2>Leaderboard</h2><p className="small">Top predictors by points and accuracy (sample).</p></div>
      {leaders.map(l=> (
        <div key={l.rank} className="card" style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
          <div><strong>#{l.rank}</strong> <span style={{marginLeft:8}}>{l.addr}</span></div>
          <div className="small">Points: {l.score}</div>
        </div>
      ))}
    </div>
  )
}
