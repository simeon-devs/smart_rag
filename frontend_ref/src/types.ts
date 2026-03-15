export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ConstraintSuggestion {
  field: string;
  label: string;
  value: string | number | string[];
  options: string[];
}

export interface ProductResult {
  product_id: string;
  source_article_id?: number;
  source_article_number?: string;
  source_l_number?: number;
  name: string;
  manufacturer?: string;
  category?: string;
  family?: string;
  price_chf?: number;
  wattage?: number;
  kelvin?: number;
  material?: string;
  style?: string;
  finish?: string;
  mood?: string;
  room_type?: string;
  image_url?: string;
  tags: string[];
  similarity_score: number;
  final_score: number;
  violations: string[];
}

export interface MemoryEntry {
  text: string;
  memory_type: string;
  source: string;
  raw_score: number;
  decay_weight: number;
  final_score: number;
  timestamp: number;
}

export interface MemoryContext {
  structural: MemoryEntry[];
  semantic: MemoryEntry[];
  episodic: MemoryEntry[];
  summary: string;
}
