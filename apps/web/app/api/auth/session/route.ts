import { NextResponse, type NextRequest } from 'next/server'

import {
  SESSION_COOKIE_NAME,
  buildSessionCookie,
  destroySessionCookieOptions,
  getAuthConfig,
  isSessionValid,
  verifyCredentials,
} from '@/lib/auth'

export async function POST(request: NextRequest) {
  const { username, password } = (await request.json().catch(() => ({}))) as {
    username?: string
    password?: string
  }

  if (!username || !password) {
    return NextResponse.json({ message: 'Missing credentials.' }, { status: 400 })
  }

  if (!(await verifyCredentials(username, password))) {
    return NextResponse.json({ message: 'Invalid credentials.' }, { status: 401 })
  }

  const cookie = await buildSessionCookie(getAuthConfig().username)
  const response = NextResponse.json({ message: 'Authenticated.' }, { status: 200 })
  response.cookies.set(cookie.name, cookie.value, cookie.options)
  return response
}

export async function DELETE() {
  const cookie = destroySessionCookieOptions()
  const response = NextResponse.json({ message: 'Signed out.' }, { status: 200 })
  response.cookies.set(cookie.name, cookie.value, cookie.options)
  return response
}

export async function GET(request: NextRequest) {
  const sessionCookie = request.cookies.get(SESSION_COOKIE_NAME)?.value
  if (!(await isSessionValid(sessionCookie))) {
    return NextResponse.json({ authenticated: false }, { status: 200 })
  }

  const { username } = getAuthConfig()
  return NextResponse.json({ authenticated: true, username }, { status: 200 })
}
