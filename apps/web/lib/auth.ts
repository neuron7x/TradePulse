export const SESSION_COOKIE_NAME = 'tp-session'
const DEFAULT_USERNAME = 'demo'
const DEFAULT_PASSWORD_PEPPER = 'tradepulse-pepper'
const DEFAULT_PASSWORD_HASH = 'c9c48a2f482a483fa73b4b32134a7be3474a76218be02277da0d794f840a058a'
const DEFAULT_SESSION_SECRET = 'tradepulse-session-secret'

const encoder = new TextEncoder()
const decoder = new TextDecoder()

function getCrypto(): Crypto {
  if (typeof globalThis.crypto !== 'undefined' && globalThis.crypto.subtle) {
    return globalThis.crypto
  }
  throw new Error('Web Crypto API is unavailable in this runtime environment')
}

function toHex(buffer: ArrayBuffer): string {
  const view = new Uint8Array(buffer)
  let hex = ''
  view.forEach((byte) => {
    hex += byte.toString(16).padStart(2, '0')
  })
  return hex
}

function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) {
    return false
  }

  let mismatch = 0
  for (let index = 0; index < a.length; index += 1) {
    mismatch |= a.charCodeAt(index) ^ b.charCodeAt(index)
  }
  return mismatch === 0
}

function encodeBase64Url(value: string): string {
  if (typeof Buffer !== 'undefined') {
    return Buffer.from(value, 'utf8').toString('base64url')
  }

  const bytes = encoder.encode(value)
  let binary = ''
  bytes.forEach((byte) => {
    binary += String.fromCharCode(byte)
  })
  if (typeof btoa === 'function') {
    const base64 = btoa(binary)
    return base64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '')
  }

  throw new Error('Base64 encoding is unavailable in this runtime environment')
}

function decodeBase64Url(value: string): string | null {
  const padded = value.padEnd(value.length + ((4 - (value.length % 4)) % 4), '=').replace(/-/g, '+').replace(/_/g, '/')

  if (typeof Buffer !== 'undefined') {
    try {
      return Buffer.from(padded, 'base64').toString('utf8')
    } catch (error) {
      return null
    }
  }

  if (typeof atob === 'function') {
    try {
      const binary = atob(padded)
      const bytes = new Uint8Array(binary.length)
      for (let index = 0; index < binary.length; index += 1) {
        bytes[index] = binary.charCodeAt(index)
      }
      return decoder.decode(bytes)
    } catch (error) {
      return null
    }
  }

  return null
}

async function sha256(input: string): Promise<string> {
  const crypto = getCrypto()
  const digest = await crypto.subtle.digest('SHA-256', encoder.encode(input))
  return toHex(digest)
}

async function hmacSha256(message: string, secret: string): Promise<string> {
  const crypto = getCrypto()
  const key = await crypto.subtle.importKey(
    'raw',
    encoder.encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign'],
  )
  const signature = await crypto.subtle.sign('HMAC', key, encoder.encode(message))
  return toHex(signature)
}

export type AuthConfig = {
  username: string
  passwordHash: string
  passwordPepper: string
  sessionSecret: string
}

export function getAuthConfig(): AuthConfig {
  return {
    username: process.env.SCENARIO_STUDIO_USERNAME?.trim() || DEFAULT_USERNAME,
    passwordHash: process.env.SCENARIO_STUDIO_PASSWORD_HASH?.trim() || DEFAULT_PASSWORD_HASH,
    passwordPepper: process.env.SCENARIO_STUDIO_PASSWORD_PEPPER?.trim() || DEFAULT_PASSWORD_PEPPER,
    sessionSecret: process.env.SCENARIO_STUDIO_SESSION_SECRET?.trim() || DEFAULT_SESSION_SECRET,
  }
}

export async function verifyCredentials(username: string, password: string): Promise<boolean> {
  const { username: expectedUsername, passwordHash, passwordPepper } = getAuthConfig()
  if (!constantTimeEqual(expectedUsername, username)) {
    return false
  }
  const submittedHash = await sha256(password + passwordPepper)
  return constantTimeEqual(passwordHash, submittedHash)
}

export type SessionPayload = {
  username: string
  signature: string
}

export async function createSession(username: string): Promise<SessionPayload> {
  const { sessionSecret } = getAuthConfig()
  const signature = await hmacSha256(username, sessionSecret)
  return { username, signature }
}

export function serializeSession(session: SessionPayload): string {
  return encodeBase64Url(`${session.username}:${session.signature}`)
}

export function parseSession(raw: string | undefined): SessionPayload | null {
  if (!raw) {
    return null
  }
  const decoded = decodeBase64Url(raw)
  if (!decoded) {
    return null
  }
  const [username, signature] = decoded.split(':')
  if (!username || !signature) {
    return null
  }
  return { username, signature }
}

export async function isSessionValid(raw: string | undefined): Promise<boolean> {
  const session = parseSession(raw)
  if (!session) {
    return false
  }
  const { username, signature } = session
  const { username: expectedUsername, sessionSecret } = getAuthConfig()
  if (!constantTimeEqual(expectedUsername, username)) {
    return false
  }
  const expectedSignature = await hmacSha256(username, sessionSecret)
  return constantTimeEqual(expectedSignature, signature)
}

export function destroySessionCookieOptions() {
  return {
    name: SESSION_COOKIE_NAME,
    value: '',
    options: {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax' as const,
      path: '/',
      maxAge: 0,
    },
  }
}

export async function buildSessionCookie(username: string) {
  const serialized = serializeSession(await createSession(username))
  return {
    name: SESSION_COOKIE_NAME,
    value: serialized,
    options: {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax' as const,
      path: '/',
      maxAge: 60 * 60 * 12,
    },
  }
}
