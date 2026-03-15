import { useState, useCallback } from 'react';
import { ChatPanel } from './components/ChatPanel';
import { ProductsPanel } from './components/ProductsPanel';
import { MemoryPanel } from './components/MemoryPanel';
import { apiExtract, apiChat, apiSaveConstraints, apiGetMemory } from './api';
import type { ChatMessage, ConstraintSuggestion, ProductResult, MemoryContext } from './types';

function App() {
  const [messages, setMessages]             = useState<ChatMessage[]>([]);
  const [chips, setChips]                   = useState<ConstraintSuggestion[]>([]);
  const [products, setProducts]             = useState<ProductResult[]>([]);
  const [memory, setMemory]                 = useState<MemoryContext | null>(null);
  const [violationCount, setViolationCount] = useState(0);
  const [isLoading, setIsLoading]           = useState(false);

  const refreshMemory = useCallback(async () => {
    try {
      const mem = await apiGetMemory();
      setMemory(mem);
    } catch {
      // memory panel stays as-is on error
    }
  }, []);

  const handleSend = useCallback(async (text: string) => {
    if (!text.trim() || isLoading) return;

    setIsLoading(true);
    setChips([]);
    setMessages(prev => [...prev, { role: 'user', content: text }]);

    try {
      // Step 1 — detect constraints, get chips
      const extract = await apiExtract(text);
      if (Array.isArray(extract.suggestions) && extract.suggestions.length > 0) {
        setChips(extract.suggestions);
      }

      // Step 2 — retrieve products and generate reply
      const chat = await apiChat(text);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: chat.llm_reply ?? 'No reply from backend.',
      }]);
      setProducts(chat.mara_results ?? []);
      setViolationCount(chat.violation_count ?? 0);

      // Step 3 — refresh memory panel
      await refreshMemory();
    } catch {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Could not reach backend. Is the server running on port 8001?',
      }]);
    } finally {
      setIsLoading(false);
    }
  }, [isLoading, refreshMemory]);

  const handleChipYes = useCallback(async (chip: ConstraintSuggestion) => {
    setChips(prev => prev.filter(c => c.field !== chip.field));
    try {
      await apiSaveConstraints(chip.field, chip.value);
      await refreshMemory();
    } catch {
      // chip already removed; ignore silently
    }
  }, [refreshMemory]);

  const handleChipSkip = useCallback((field: string) => {
    setChips(prev => prev.filter(c => c.field !== field));
  }, []);

  return (
    <div className="flex h-screen bg-[#0a0a0f] text-gray-100 overflow-hidden">
      <ChatPanel
        messages={messages}
        chips={chips}
        isLoading={isLoading}
        violationCount={violationCount}
        onSend={handleSend}
        onChipYes={handleChipYes}
        onChipSkip={handleChipSkip}
      />
      <ProductsPanel products={products} />
      <MemoryPanel memory={memory} />
    </div>
  );
}

export default App;
