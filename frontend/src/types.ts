export interface SearchResult {
  id: number
  score: number
  label: string
  filename: string
  image_url: string
}

export interface SearchResponse {
  query: string
  k: number
  results: {
    hnsw?: SearchResult[]
    brute_force?: SearchResult[]
  }
  latency_ms: {
    hnsw?: number
    brute_force?: number
  }
}
