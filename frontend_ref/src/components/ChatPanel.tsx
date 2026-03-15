import { useState, useRef, useEffect } from 'react';
import type { ChatMessage, ConstraintSuggestion } from '../types';

interface Props {
  messages: ChatMessage[];
  chips: ConstraintSuggestion[];
  isLoading: boolean;
  violationCount: number;
  onSend: (text: string) => void;
  onChipYes: (chip: ConstraintSuggestion) => void;
  onChipSkip: (field: string) => void;
}

export function ChatPanel({
  messages, chips, isLoading, violationCount, onSend, onChipYes, onChipSkip,
}: Props) {
  const [input, setInput] = useState('');
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, chips, isLoading]);

  const handleSend = () => {
    const text = input.trim();
    if (!text || isLoading) return;
    setInput('');
    onSend(text);
  };

  return (
    <div className="w-80 flex-shrink-0 flex flex-col border-r border-[#1e1e2e] bg-[#0a0a0f]">

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#1e1e2e]">
        <div>
          <div className="text-xs font-semibold uppercase tracking-widest text-amber-400">MARA</div>
          <div className="text-[10px] text-gray-600 mt-0.5">demo_user</div>
        </div>
        {violationCount > 0 && (
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] text-gray-600">violations</span>
            <span className="w-5 h-5 rounded-full bg-red-500 text-white text-[10px] font-bold flex items-center justify-center leading-none">
              {violationCount}
            </span>
          </div>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-3 py-3 flex flex-col gap-2 min-h-0">
        {messages.length === 0 && (
          <div className="text-center mt-10 px-4">
            <div className="text-3xl mb-3 opacity-30">💡</div>
            <p className="text-[11px] text-gray-600 leading-relaxed">
              Ask about lighting
            </p>
            <p className="text-[11px] text-amber-600/50 mt-1 leading-relaxed">
              "warm light for bedroom under 200 CHF, no plastic"
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] rounded-2xl px-3 py-2 text-[11.5px] leading-relaxed ${
              msg.role === 'user'
                ? 'bg-amber-500 text-black font-medium rounded-br-sm'
                : 'bg-[#12121a] text-gray-300 rounded-bl-sm border border-[#1e1e2e]'
            }`}>
              {msg.content}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-[#12121a] border border-[#1e1e2e] rounded-2xl rounded-bl-sm px-3 py-2.5">
              <div className="flex gap-1 items-center">
                {[0, 1, 2].map(i => (
                  <span
                    key={i}
                    className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-bounce"
                    style={{ animationDelay: `${i * 0.15}s` }}
                  />
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Constraint chips */}
        {chips.length > 0 && (
          <div className="flex flex-col gap-1.5 mt-1">
            <p className="text-[10px] text-gray-600 uppercase tracking-wider px-0.5">
              Detected constraints
            </p>
            {chips.map(chip => (
              <div
                key={chip.field}
                className="bg-[#12121a] border border-amber-500/25 rounded-xl px-3 py-2.5"
              >
                <p className="text-[11px] text-gray-300 leading-snug mb-2">{chip.label}</p>
                <div className="flex gap-1.5 items-center">
                  <button
                    onClick={() => onChipYes(chip)}
                    className="px-3 py-1 rounded-lg bg-amber-500 text-black text-[11px] font-semibold hover:bg-amber-400 transition-colors"
                  >
                    Yes
                  </button>
                  <button
                    onClick={() => onChipSkip(chip.field)}
                    className="px-3 py-1 rounded-lg bg-[#1e1e2e] text-gray-500 text-[11px] hover:text-gray-300 transition-colors"
                  >
                    Skip
                  </button>
                  <span className="text-[10px] text-gray-700 ml-1 font-mono">{chip.field}</span>
                </div>
              </div>
            ))}
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="border-t border-[#1e1e2e] px-3 py-2.5 flex gap-2 items-center">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSend()}
          placeholder="Ask about lighting…"
          disabled={isLoading}
          className="flex-1 bg-[#12121a] border border-[#1e1e2e] rounded-xl px-3 py-2 text-[12px] text-gray-200 placeholder:text-gray-700 outline-none focus:border-amber-500/40 transition-colors disabled:opacity-40"
        />
        <button
          onClick={handleSend}
          disabled={isLoading || !input.trim()}
          className="w-8 h-8 rounded-xl bg-amber-500 text-black flex items-center justify-center hover:bg-amber-400 transition-colors disabled:opacity-30 flex-shrink-0"
          aria-label="Send"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <path d="M22 2L11 13M22 2L15 22L11 13L2 9L22 2Z" />
          </svg>
        </button>
      </div>
    </div>
  );
}
