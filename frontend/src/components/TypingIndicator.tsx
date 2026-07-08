import { useEffect, useState } from 'react';

export default function TypingIndicator() {
  const [dots, setDots] = useState('');
  useEffect(() => {
    const interval = setInterval(() => {
      setDots((d) => (d.length >= 3 ? '' : d + '.'));
    }, 400);
    return () => clearInterval(interval);
  }, []);
  return (
    <div className="flex justify-start px-6 py-3">
      <div className="max-w-[75%] rounded-2xl rounded-bl-md px-4 py-3 bg-black/40 backdrop-blur-md border border-white/5">
        <div className="flex items-center gap-2">
          <div className="flex gap-1">
            <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
            <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
            <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
          </div>
          <span className="text-sm text-gray-400">Thinking{dots}</span>
        </div>
      </div>
    </div>
  );
}
