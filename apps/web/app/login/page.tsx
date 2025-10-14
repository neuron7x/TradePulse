'use client'

import { useRouter, useSearchParams } from 'next/navigation'
import type { CSSProperties, FormEvent, ReactNode } from 'react'
import { Suspense, useCallback, useMemo, useState } from 'react'

const layoutStyle: CSSProperties = {
  minHeight: '100vh',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  background: 'radial-gradient(circle at top, #0f172a 0%, #020617 70%)',
  color: '#e2e8f0',
  padding: '2rem',
}

const panelStyle: CSSProperties = {
  width: '100%',
  maxWidth: '420px',
  backgroundColor: 'rgba(15, 23, 42, 0.85)',
  borderRadius: '1rem',
  border: '1px solid rgba(148, 163, 184, 0.25)',
  padding: '2rem',
  display: 'grid',
  gap: '1.25rem',
  boxShadow: '0 35px 60px -40px rgba(15, 23, 42, 0.7)',
}

const labelStyle: CSSProperties = {
  display: 'block',
  fontWeight: 600,
  marginBottom: '0.5rem',
}

const inputStyle: CSSProperties = {
  width: '100%',
  padding: '0.75rem 0.85rem',
  borderRadius: '0.75rem',
  border: '1px solid rgba(148, 163, 184, 0.3)',
  backgroundColor: '#020c1b',
  color: '#f8fafc',
  fontSize: '1rem',
}

const buttonStyle: CSSProperties = {
  width: '100%',
  padding: '0.85rem 1rem',
  borderRadius: '0.75rem',
  border: 'none',
  background: 'linear-gradient(135deg, #22d3ee 0%, #0ea5e9 60%)',
  color: '#0f172a',
  fontWeight: 700,
  fontSize: '1.05rem',
  cursor: 'pointer',
  transition: 'filter 120ms ease',
}

const hintStyle: CSSProperties = {
  fontSize: '0.85rem',
  color: '#94a3b8',
  lineHeight: 1.5,
}

const errorStyle: CSSProperties = {
  fontSize: '0.9rem',
  color: '#f97316',
  fontWeight: 600,
  textAlign: 'center',
}

function LoginForm() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const redirectTo = useMemo(() => searchParams?.get('next') ?? '/', [searchParams])

  const handleSubmit = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault()
      setError(null)
      setSubmitting(true)

      try {
        const response = await fetch('/api/auth/session', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, password }),
        })

        if (!response.ok) {
          const payload = (await response.json().catch(() => ({}))) as { message?: string }
          const message = payload.message ?? 'Unable to authenticate with the provided credentials.'
          setError(message)
          setSubmitting(false)
          return
        }

        setSubmitting(false)
        router.replace(redirectTo)
      } catch (requestError) {
        setSubmitting(false)
        setError('Unexpected error while reaching the authentication service. Please retry.')
      }
    },
    [username, password, redirectTo, router],
  )

  return (
    <main style={layoutStyle}>
      <section style={panelStyle}>
        <header style={{ textAlign: 'center', display: 'grid', gap: '0.5rem' }}>
          <h1 style={{ margin: 0, fontSize: '2rem' }}>TradePulse Scenario Studio</h1>
          <p style={{ ...hintStyle, margin: 0 }}>
            Access is restricted to authorised analysts. Provide the credentials distributed by the operations team.
          </p>
        </header>

        <form onSubmit={handleSubmit} style={{ display: 'grid', gap: '1.5rem' }}>
          <div>
            <label htmlFor="username" style={labelStyle}>
              Username
            </label>
            <input
              id="username"
              name="username"
              autoComplete="username"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              style={inputStyle}
              placeholder="Enter your username"
              required
              disabled={submitting}
            />
          </div>

          <div>
            <label htmlFor="password" style={labelStyle}>
              Password
            </label>
            <input
              id="password"
              name="password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              style={inputStyle}
              placeholder="Enter your password"
              required
              disabled={submitting}
            />
          </div>

          <button
            type="submit"
            style={{ ...buttonStyle, filter: submitting ? 'grayscale(35%)' : undefined }}
            disabled={submitting}
          >
            {submitting ? 'Verifying…' : 'Sign in'}
          </button>
        </form>

        <p style={hintStyle}>
          Strongly recommend using individual credentials and rotating them every 90 days. Contact the security team for
          assistance with password resets or MFA device provisioning.
        </p>

        {error ? (
          <div role="alert" aria-live="assertive" style={errorStyle}>
            {error}
          </div>
        ) : null}
      </section>
    </main>
  )
}

export default function LoginPage(): ReactNode {
  return (
    <Suspense
      fallback={
        <main style={layoutStyle}>
          <section style={panelStyle}>
            <h1 style={{ margin: 0, fontSize: '1.5rem', textAlign: 'center' }}>Loading sign-in…</h1>
          </section>
        </main>
      }
    >
      <LoginForm />
    </Suspense>
  )
}
