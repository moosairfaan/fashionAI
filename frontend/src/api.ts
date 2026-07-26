import type { SearchResponse } from './types'

const API = 'http://localhost:8000'

export async function search(query: string, k = 12): Promise<SearchResponse> {
  const res = await fetch(`${API}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, k, method: 'both' }),
  })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(detail || `Search failed (${res.status})`)
  }
  return res.json()
}

/** Backend returns paths like /images/... — point them at the API host. */
export function imageSrc(imageUrl: string): string {
  if (imageUrl.startsWith('http')) return imageUrl
  return `${API}${imageUrl}`
}
