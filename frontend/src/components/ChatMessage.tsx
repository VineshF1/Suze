import ReactMarkdown from 'react-markdown';
import type { Message } from '../types';

interface ChatMessageProps {
  message: Message;
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex px-6 py-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[75%] rounded-2xl px-4 py-3 backdrop-blur-md border border-white/5 ${
        isUser
          ? 'bg-black/50 text-white rounded-br-md'
          : 'bg-black/40 text-white rounded-bl-md'
      }`}>
        <div className="prose prose-sm max-w-none leading-relaxed">
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>
        {message.sources && message.sources.length > 0 && (
          <details className="mt-2 hidden">
            <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-600">
              Sources ({message.sources.length})
            </summary>
          </details>
        )}
      </div>
    </div>
  );
}
