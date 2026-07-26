import { useEffect, useRef, useState, type FormEvent } from 'react'
import { imageSrc, search } from './api'
import type { SearchResponse, SearchResult } from './types'

/** Count from 0 → target over ~400ms when `target` / `token` changes. */
function CountUp({
  target,
  decimals = 2,
  suffix = '',
  token,
}: {
  target: number
  decimals?: number
  suffix?: string
  token: number
}) {
  const [value, setValue] = useState(0)
  const raf = useRef(0)

  useEffect(() => {
    cancelAnimationFrame(raf.current)
    const duration = 400
    const start = performance.now()
    const from = 0
    const to = target

    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration)
      // ease-out cubic
      const eased = 1 - (1 - t) ** 3
      setValue(from + (to - from) * eased)
      if (t < 1) raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf.current)
  }, [target, token])

  return (
    <span className="font-mono tabular-nums">
      {value.toFixed(decimals)}
      {suffix}
    </span>
  )
}

function ResultCard({ item }: { item: SearchResult }) {
  return (
    <figure className="group relative overflow-hidden">
      <div className="aspect-[3/4] overflow-hidden bg-[#e8e4dc]">
        <img
          src={imageSrc(item.image_url)}
          alt={item.label || item.filename}
          className="h-full w-full object-cover transition-transform duration-200 ease-out group-hover:scale-[1.02]"
          loading="lazy"
        />
      </div>
      <figcaption className="pointer-events-none absolute bottom-2 left-2 font-mono text-[11px] text-accent opacity-0 transition-opacity duration-200 group-hover:opacity-100">
        {(item.score * 100).toFixed(1)}%
      </figcaption>
    </figure>
  )
}

function ResultColumn({
  title,
  count,
  items,
}: {
  title: string
  count: number
  items: SearchResult[]
}) {
  return (
    <section className="min-w-0 flex-1">
      <div className="mb-5 flex items-baseline gap-3">
        <h2 className="font-sans text-[11px] font-medium tracking-[0.14em] text-muted uppercase">
          {title}
        </h2>
        <span className="font-mono text-[11px] text-muted tabular-nums">{count}</span>
      </div>
      <div className="grid grid-cols-2 gap-6 sm:grid-cols-3">
        {items.map((item) => (
          <ResultCard key={`${title}-${item.id}`} item={item} />
        ))}
      </div>
    </section>
  )
}

export default function App() {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<SearchResponse | null>(null)
  const [animToken, setAnimToken] = useState(0)

  async function onSearch(e?: FormEvent) {
    e?.preventDefault()
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    try {
      const res = await search(query.trim(), 12)
      setData(res)
      setAnimToken((t) => t + 1)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed')
      setData(null)
    } finally {
      setLoading(false)
    }
  }

  const hnswMs = data?.latency_ms.hnsw ?? 0
  const bruteMs = data?.latency_ms.brute_force ?? 0
  const speedup = hnswMs > 0 ? bruteMs / hnswMs : 0

  return (
    <div className="min-h-screen px-5 py-10 sm:px-8 md:px-12">
      <header className="mb-12">
        <h1 className="font-serif text-3xl font-normal tracking-[0.04em] text-ink sm:text-4xl">
          fashionAI
        </h1>
      </header>

      <form onSubmit={onSearch} className="flex items-stretch gap-3">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="search the collection"
          className="min-w-0 flex-1 rounded-[4px] border border-ink/20 bg-transparent px-4 py-3.5 font-serif text-lg font-normal tracking-[0.02em] text-ink placeholder:text-muted/80 outline-none transition-colors focus:border-accent"
          aria-label="Search the collection"
        />
        <button
          type="submit"
          disabled={loading}
          className="rounded-[4px] border border-ink bg-ink px-5 py-3.5 font-sans text-xs font-medium tracking-[0.12em] text-paper uppercase transition-colors hover:bg-accent hover:border-accent disabled:opacity-40"
        >
          {loading ? '…' : 'Search'}
        </button>
      </form>

      {error && (
        <p className="mt-4 font-sans text-sm text-accent">{error}</p>
      )}

      {data && (
        <div className="mt-8">
          {/* Latency readout — flat bar, not a card */}
          <div className="flex flex-wrap items-center gap-x-0 border-y border-ink/10 py-3 text-[11px] text-muted">
            <div className="flex items-baseline gap-2 pr-4 sm:pr-5">
              <span className="font-sans tracking-[0.08em] uppercase">HNSW</span>
              <span className="text-ink">
                <CountUp target={hnswMs} decimals={2} suffix="ms" token={animToken} />
              </span>
            </div>
            <div className="hidden h-3 w-px bg-ink/15 sm:block" aria-hidden />
            <div className="flex items-baseline gap-2 px-4 sm:px-5">
              <span className="font-sans tracking-[0.08em] uppercase">Brute Force</span>
              <span className="text-ink">
                <CountUp target={bruteMs} decimals={2} suffix="ms" token={animToken} />
              </span>
            </div>
            <div className="hidden h-3 w-px bg-ink/15 sm:block" aria-hidden />
            <div className="flex items-baseline gap-2 pl-4 sm:pl-5">
              <span className="font-sans tracking-[0.08em] uppercase">Speedup</span>
              <span className="text-accent">
                <CountUp target={speedup} decimals={2} suffix="x" token={animToken} />
              </span>
            </div>
          </div>

          <div className="mt-10 flex flex-col gap-14 lg:flex-row lg:gap-10">
            <ResultColumn
              title="HNSW Results"
              count={data.results.hnsw?.length ?? 0}
              items={data.results.hnsw ?? []}
            />
            <ResultColumn
              title="Brute Force Results"
              count={data.results.brute_force?.length ?? 0}
              items={data.results.brute_force ?? []}
            />
          </div>
        </div>
      )}
    </div>
  )
}
