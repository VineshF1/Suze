import { useState, useRef, useEffect } from 'react';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import TypingIndicator from './components/TypingIndicator';
import { Vortex } from './components/ui/vortex';
import { chat, ingestFile, healthCheck } from './api';
import type { Message } from './types';

function SuzeLogo() {
  return (
    <img src="/logo.png" alt="Suze" className="h-8 w-auto rounded-md" />
  );
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [backendOnline, setBackendOnline] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  // Check backend health on mount
  useEffect(() => {
    healthCheck()
      .then(() => setBackendOnline(true))
      .catch(() => setBackendOnline(false));
  }, []);

  const handleToggleBackend = () => {
    if (backendOnline) {
      setBackendOnline(false);
    } else {
      healthCheck()
        .then(() => setBackendOnline(true))
        .catch(() => setBackendOnline(false));
    }
  };

  const handleNewChat = () => {
    setMessages([]);
    setError(null);
  };

  const handleSend = async (text: string) => {
    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: text,
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);
    setError(null);

    try {
      const res = await chat(text);
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.answer || 'No answer generated.',
        sources: data.sources,
        trace: data.trace,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: any) {
      setError(err.message || 'Failed to get response');
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: `Error: ${err.message || 'Something went wrong. Please try again.'}`,
          timestamp: Date.now(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelected = async (file: File) => {
    setLoading(true);
    setError(null);
    try {
      const result = await ingestFile(file);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: 'user',
          content: `📄 Uploaded: **${file.name}** (${result.chunks_ingested || 0} chunks ingested)`,
          timestamp: Date.now(),
        },
        {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: `✅ **${file.name}** has been ingested. Total chunks: ${result.total_chunks || 0}. You can now ask questions about it.`,
          timestamp: Date.now(),
        },
      ]);
    } catch (err: any) {
      setError(err.message || 'Failed to upload file');
    } finally {
      setLoading(false);
    }
  };

  const isHome = messages.length === 0 && !loading;

  return (
    <Vortex
      particleCount={500}
      baseHue={60}
      rangeY={800}
      backgroundColor="#000000"
      containerClassName="h-screen"
      className="flex flex-col h-full text-gray-100"
    >
      {/* Header */}
      <header className="relative z-10 flex items-center justify-between px-6 py-4">
        <div className="flex items-center gap-2">
          <SuzeLogo />
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={handleNewChat}
            className="p-2 rounded-lg text-gray-400 hover:text-gray-200 hover:bg-[#1a1a1d] transition-colors cursor-pointer"
            title="New chat"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 2v6h-6" />
              <path d="M3 12a9 9 0 0 1 15-6.7L21 8" />
              <path d="M3 12a9 9 0 0 0 15 6.7L21 16" />
            </svg>
          </button>
          <button
            onClick={handleToggleBackend}
            className="relative p-1.5 rounded-full transition-colors cursor-pointer"
            title={backendOnline ? 'Backend online' : 'Backend offline'}
          >
            <span
              className={`block w-3 h-3 rounded-full animate-pulse ${
                backendOnline
                  ? 'bg-green-400 shadow-[0_0_8px_rgba(74,222,128,0.6)]'
                  : 'bg-red-400 shadow-[0_0_8px_rgba(248,113,113,0.6)]'
              }`}
            />
          </button>
        </div>
      </header>

      {/* Main content */}
      <main className="relative z-10 flex-1 flex flex-col min-w-0">
        {isHome ? (
          /* ── Welcome screen ── */
          <div className="flex-1 flex flex-col items-center justify-center px-4">
            <h1 className="text-5xl md:text-6xl font-bold text-white tracking-tight">
              Suze
            </h1>
            <p className="text-sm text-gray-500 mt-3 max-w-md text-center">
              Secure Agentic RAG Knowledge Assistant.
            </p>
          </div>
        ) : (
          /* ── Chat messages ── */
          <div className="flex-1 overflow-y-auto">
            {messages.map((msg) => (
              <ChatMessage key={msg.id} message={msg} />
            ))}
            {loading && <TypingIndicator />}
            {error && !loading && (
              <div className="px-4 py-2 text-center">
                <span className="text-xs text-red-400 bg-red-400/10 px-3 py-1 rounded-full">{error}</span>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>
        )}

        {/* Input area */}
        <div className={`relative z-10 ${isHome ? 'pb-8' : 'pb-4 pt-4'}`}>
          <div className="px-4 max-w-2xl mx-auto">
            <ChatInput onSend={handleSend} onFileSelected={handleFileSelected} disabled={loading} />
            <p className="text-center text-xs text-gray-600 mt-3">
              Suze may occasionally display inaccurate answers. Please verify.
            </p>
          </div>
        </div>
      </main>
    </Vortex>
  );
}

export default App;
