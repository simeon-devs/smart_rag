import type { ProductResult } from '../types';

interface Props {
  products: ProductResult[];
}

export function ProductsPanel({ products }: Props) {
  return (
    <div className="flex-1 flex flex-col min-w-0 border-r border-[#1e1e2e]">

      {/* Header */}
      <div className="px-4 py-3 border-b border-[#1e1e2e] flex items-center justify-between flex-shrink-0">
        <span className="text-xs font-semibold uppercase tracking-widest text-gray-500">
          Products
        </span>
        {products.length > 0 && (
          <span className="text-[11px] text-amber-400 font-medium">
            {products.length} result{products.length !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      {/* Grid */}
      <div className="flex-1 overflow-y-auto p-3 min-h-0">
        {products.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-700 gap-2">
            <span className="text-4xl opacity-20">🔦</span>
            <p className="text-[11px]">Products appear here after a search</p>
          </div>
        ) : (
          <div className="grid gap-3"
            style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))' }}
          >
            {products.map(p => (
              <ProductCard key={p.product_id} product={p} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ProductCard({ product: p }: { product: ProductResult }) {
  const clean = p.violations.length === 0;

  return (
    <div className={`bg-[#12121a] rounded-xl overflow-hidden border flex flex-col ${
      clean ? 'border-[#1e1e2e]' : 'border-red-900/40'
    }`}>

      {/* Image */}
      <div className="w-full h-28 bg-[#0d0d18] flex items-center justify-center overflow-hidden flex-shrink-0">
        {p.image_url ? (
          <img
            src={p.image_url}
            alt={p.name}
            className="w-full h-full object-contain p-2"
            onError={e => {
              const el = e.target as HTMLImageElement;
              el.style.display = 'none';
              el.parentElement!.innerHTML = '<span style="font-size:28px;opacity:0.15">💡</span>';
            }}
          />
        ) : (
          <span className="text-3xl opacity-15">💡</span>
        )}
      </div>

      {/* Info */}
      <div className="p-2.5 flex flex-col gap-1.5 flex-1">
        <p className="text-[11px] text-gray-200 font-medium leading-snug line-clamp-2">{p.name}</p>

        {/* Specs row */}
        <div className="flex flex-wrap gap-x-2 gap-y-0.5">
          {p.price_chf != null && (
            <span className="text-[11px] text-amber-400 font-semibold">{p.price_chf} CHF</span>
          )}
          {p.wattage != null && (
            <span className="text-[10px] text-gray-500">{p.wattage}W</span>
          )}
          {p.kelvin != null && (
            <span className="text-[10px] text-gray-500">{p.kelvin}K</span>
          )}
        </div>

        {/* Score + status */}
        <div className="flex items-center justify-between mt-auto pt-1.5 border-t border-[#1e1e2e]">
          <span className="text-[10px] text-gray-700 font-mono">
            {p.final_score.toFixed(3)}
          </span>
          {clean ? (
            <span className="flex items-center gap-0.5 text-[10px] text-emerald-400">
              <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                <polyline points="20 6 9 17 4 12" />
              </svg>
              OK
            </span>
          ) : (
            <span className="flex items-center gap-0.5 text-[10px] text-red-400">
              <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
              {p.violations.length}
            </span>
          )}
        </div>

        {/* Violation detail */}
        {p.violations.length > 0 && (
          <div className="border-t border-red-900/25 pt-1">
            {p.violations.map((v, i) => (
              <p key={i} className="text-[10px] text-red-400/70 leading-tight">{v}</p>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
