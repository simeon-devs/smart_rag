import type { MemoryContext, MemoryEntry } from '../types';

interface Props {
  memory: MemoryContext | null;
}

export function MemoryPanel({ memory }: Props) {
  return (
    <div className="w-64 flex-shrink-0 flex flex-col bg-[#0a0a0f]">

      {/* Header */}
      <div className="px-4 py-3 border-b border-[#1e1e2e]">
        <div className="text-xs font-semibold uppercase tracking-widest text-gray-500">
          Memory
        </div>
        {memory && (
          <div className="flex gap-3 mt-1.5">
            <CountBadge label="Hard" count={memory.structural.length} color="text-blue-400" />
            <CountBadge label="Pref" count={memory.semantic.length} color="text-purple-400" />
            <CountBadge label="Epis" count={memory.episodic.length} color="text-amber-400" />
          </div>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-4 min-h-0">
        {!memory ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-700 gap-2 mt-8">
            <span className="text-2xl opacity-20">🧠</span>
            <p className="text-[11px] text-center leading-relaxed">
              Memory updates after<br />first message
            </p>
          </div>
        ) : (
          <>
            <MemorySection
              title="Hard Constraints"
              entries={memory.structural}
              accentClass="text-blue-400"
              emptyText="No constraints saved"
            />
            <MemorySection
              title="Preferences"
              entries={memory.semantic}
              accentClass="text-purple-400"
              emptyText="No preferences learned"
            />
            <MemorySection
              title="Recent"
              entries={memory.episodic}
              accentClass="text-amber-400"
              emptyText="No recent activity"
            />
          </>
        )}
      </div>
    </div>
  );
}

function CountBadge({ label, count, color }: { label: string; count: number; color: string }) {
  return (
    <div className="flex items-center gap-1">
      <span className={`text-[10px] font-semibold ${color}`}>{label}</span>
      <span className="text-[10px] text-gray-600">{count}</span>
    </div>
  );
}

function MemorySection({
  title, entries, accentClass, emptyText,
}: {
  title: string;
  entries: MemoryEntry[];
  accentClass: string;
  emptyText: string;
}) {
  return (
    <div>
      <div className={`text-[10px] font-semibold uppercase tracking-wider mb-2 ${accentClass}`}>
        {title}
      </div>
      {entries.length === 0 ? (
        <p className="text-[10px] text-gray-700 italic">{emptyText}</p>
      ) : (
        <div className="flex flex-col gap-1.5">
          {entries.map((entry, i) => (
            <MemoryRow key={i} entry={entry} />
          ))}
        </div>
      )}
    </div>
  );
}

function MemoryRow({ entry }: { entry: MemoryEntry }) {
  const opacity = Math.max(0.25, entry.decay_weight);
  const pct = Math.round(entry.decay_weight * 100);

  return (
    <div className="flex items-start justify-between gap-2" style={{ opacity }}>
      <p className="text-[11px] text-gray-300 leading-snug flex-1">{entry.text}</p>
      <span className="text-[9px] text-gray-600 flex-shrink-0 mt-0.5 font-mono">{pct}%</span>
    </div>
  );
}
