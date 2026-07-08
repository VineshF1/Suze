import { useState, useRef, useEffect } from 'react';
import { Send } from 'lucide-react';

interface ChatInputProps {
  onSend: (text: string) => void;
  onFileSelected?: (file: File) => void;
  disabled: boolean;
}

export default function ChatInput({ onSend, onFileSelected, disabled }: ChatInputProps) {
  const [text, setText] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = () => {
    if (!text.trim() || disabled) return;
    onSend(text.trim());
    setText('');
    inputRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const hasText = text.trim().length > 0;

  return (
    <>
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,.md,.txt"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file && onFileSelected) {
            onFileSelected(file);
            e.target.value = '';
          }
        }}
      />
      <div className="flex items-center gap-2 bg-black rounded-full px-4 py-3 border border-[#1e1e21] focus-within:border-[#3a3a3d] transition-colors w-full max-w-2xl mx-auto">
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
          className="flex-shrink-0 p-1 rounded-full text-gray-500 hover:text-gray-300 hover:bg-[#1e1e21] transition-colors cursor-pointer"
          title="Upload document"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 5v14" />
            <path d="M5 12h14" />
          </svg>
        </button>
        <input
          ref={inputRef}
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder=""
          disabled={disabled}
          className="flex-1 bg-transparent text-gray-100 outline-none text-sm"
        />
        <button
          onClick={handleSubmit}
          disabled={!hasText || disabled}
          className={`flex-shrink-0 p-2 rounded-full transition-all cursor-pointer ${
            hasText && !disabled
              ? 'bg-white text-black hover:bg-gray-200 ring-2 ring-white/40 shadow-[0_0_14px_rgba(255,255,255,0.2)]'
              : 'text-gray-500 ring-1 ring-white/10'
          }`}
        >
          <Send size={16} />
        </button>
      </div>
    </>
  );
}
