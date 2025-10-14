import { NextResponse, type NextRequest } from 'next/server'

import { SESSION_COOKIE_NAME, isSessionValid } from '@/lib/auth'

const PUBLIC_PATHS = new Set(['/login', '/api/auth/session'])

function isPublicPath(pathname: string): boolean {
  if (PUBLIC_PATHS.has(pathname)) {
    return true
  }

  if (pathname.startsWith('/_next')) {
    return true
  }

  if (pathname.startsWith('/static') || pathname.startsWith('/assets')) {
    return true
  }

  return false
}

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl
  const sessionCookie = request.cookies.get(SESSION_COOKIE_NAME)?.value
  const hasSession = await isSessionValid(sessionCookie)

  if (isPublicPath(pathname)) {
    if (pathname === '/login' && hasSession) {
      const redirectUrl = new URL('/', request.url)
      return NextResponse.redirect(redirectUrl)
    }
    return NextResponse.next()
  }

  if (!hasSession) {
    const redirectUrl = new URL('/login', request.url)
    return NextResponse.redirect(redirectUrl)
  }

  return NextResponse.next()
}

export const config = {
  matcher: ['/((?!api/auth/session).*)'],
}
