import type { Metadata } from 'next'
import type { ReactNode } from 'react'

export const metadata: Metadata = {
  title: 'TradePulse Scenario Studio',
  description: 'Sanity-check strategy templates with guardrails before promoting them to production.',
}

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          backgroundColor: '#020617',
          color: '#e2e8f0',
          fontFamily:
            "'Inter', 'Segoe UI', 'Helvetica Neue', system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
        }}
      >
        {children}
      </body>
    </html>
  )
}
