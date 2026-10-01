// Web Push on the phone (design §8.3). iOS only delivers notifications to Degas once
// it's installed to the home screen, and only asks for permission from a tap.

import { api } from '@/api/client'
import { writeStored } from '@/lib/storage'

const ASKED_KEY = 'degas.push.asked'

export type PushState =
  | 'unsupported' // no service worker or Push API (or not installed, on iOS)
  | 'blocked' // the user refused permission
  | 'off'
  | 'on'

/** Running as an installed home-screen app. */
export function isInstalled(): boolean {
  const nav = navigator as Navigator & { standalone?: boolean }
  return (
    nav.standalone === true ||
    (typeof matchMedia === 'function' && matchMedia('(display-mode: standalone)').matches)
  )
}

export function pushSupported(): boolean {
  return 'serviceWorker' in navigator && 'PushManager' in window && 'Notification' in window
}

async function registration(): Promise<ServiceWorkerRegistration> {
  return navigator.serviceWorker.ready
}

export async function pushState(): Promise<PushState> {
  if (!pushSupported()) return 'unsupported'
  if (Notification.permission === 'denied') return 'blocked'
  // No worker registered (the dev server): nothing could deliver a notification.
  const reg = await navigator.serviceWorker.getRegistration()
  if (!reg) return 'unsupported'
  const sub = await reg.pushManager.getSubscription()
  return sub && Notification.permission === 'granted' ? 'on' : 'off'
}

/** Ask for permission (call from a tap) and subscribe this device. */
export async function enablePush(): Promise<PushState> {
  markAsked()
  const permission = await Notification.requestPermission()
  if (permission !== 'granted') return permission === 'denied' ? 'blocked' : 'off'
  const reg = await registration()
  const { public_key } = await api.pushKey()
  const sub =
    (await reg.pushManager.getSubscription()) ??
    (await reg.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: base64UrlBytes(public_key),
    }))
  await api.pushSubscribe(sub.toJSON())
  return 'on'
}

export async function disablePush(): Promise<PushState> {
  const sub = await (await registration()).pushManager.getSubscription()
  if (sub) {
    await api.pushUnsubscribe(sub.endpoint)
    await sub.unsubscribe()
  }
  return 'off'
}

/** Whether Degas has already asked (it asks once; after that it's a switch in the sheet). */
export function wasAsked(): boolean {
  try {
    return localStorage.getItem(ASKED_KEY) !== null
  } catch {
    return true // can't remember the answer, so don't keep asking
  }
}

export function markAsked() {
  writeStored(ASKED_KEY, new Date().toISOString())
}

function base64UrlBytes(value: string): Uint8Array<ArrayBuffer> {
  const b64 = value.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat((4 - (value.length % 4)) % 4)
  const raw = atob(b64)
  const out = new Uint8Array(raw.length)
  for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i)
  return out
}
