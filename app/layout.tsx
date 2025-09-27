import './globals.css'
import Link from 'next/link'

export const metadata = {
  title: 'PrediXchain (Sample)',
  description: 'Student Forex Oracle - sample Next.js app',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="site-header">
          <div className="container">
            <h1 className="brand"><Link href="/">PrediXchain</Link></h1>
            <nav>
              <Link href="/">Home</Link>
              <Link href="/predictions">Prediction Feed</Link>
              <Link href="/leaderboard">Leaderboard</Link>
              <Link href="/rewards">Rewards</Link>
            </nav>
          </div>
        </header>
        <main className="container main">{children}</main>
        <footer className="site-footer">
          <div className="container">Built for a hackathon — sample frontend. © PrediXchain</div>
        </footer>
      </body>
    </html>
  )
}
