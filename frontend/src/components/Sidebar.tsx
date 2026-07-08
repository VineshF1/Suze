import { Menu, X } from 'lucide-react';

interface SidebarProps {
  open: boolean;
  onToggle: () => void;
}

export default function Sidebar({ open, onToggle }: SidebarProps) {
  return (
    <>
      {/* Floating toggle button — always visible when sidebar is closed */}
      {!open && (
        <button
          onClick={onToggle}
          className="fixed right-4 top-1/2 -translate-y-1/2 z-50 w-11 h-11 rounded-full bg-[#1a1a1d] border border-[#2a2a2d] flex items-center justify-center text-gray-400 hover:text-gray-200 hover:bg-[#2a2a2d] transition-all cursor-pointer shadow-lg"
        >
          <Menu size={18} />
        </button>
      )}

      {/* Overlay when open on mobile */}
      {open && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={onToggle}
        />
      )}

      {/* Sidebar panel — slides from right */}
      <div
        className={`fixed top-0 right-0 h-full z-50 w-72 bg-[#1a1a1d] border-l border-[#2a2a2d] flex flex-col transition-transform duration-300 ease-in-out ${
          open ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        {/* Header */}
        <div className="flex items-center justify-end p-4 border-b border-[#2a2a2d]">
          <button
            onClick={onToggle}
            className="p-1.5 rounded-lg text-gray-400 hover:text-gray-200 hover:bg-[#2a2a2d] transition-colors cursor-pointer"
          >
            <X size={18} />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          <p className="text-xs text-gray-500 text-center py-8">No chat history yet.</p>
        </div>
      </div>
    </>
  );
}
