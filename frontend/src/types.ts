export interface Source {
  text: string;
  source: string;
  score: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  trace?: any[];
  timestamp: number;
}

export interface ChatResponse {
  query: string;
  answer: string;
  sources: Source[];
  trace: any[];
}

export interface IngestResponse {
  chunks_ingested: number;
  total_chunks: number;
  message: string;
}
