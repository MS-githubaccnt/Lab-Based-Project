import React from 'react';
import { motion } from 'framer-motion';
import { Trash2, MessageSquare } from 'lucide-react';

interface ListButtonProps {
  id: number;
  title: string;
  isActive: boolean;
  onClick: (id: number) => void;
  onDelete: (id: number) => void;
}

export const ListButton: React.FC<ListButtonProps> = ({ id, title, isActive, onClick, onDelete }) => {
  return (
    <motion.div
      layout
      variants={{
        hidden: { opacity: 0, x: -20 },
        visible: { opacity: 1, x: 0 }
      }}
      initial="hidden"
      animate="visible"
      exit={{ opacity: 0, x: -50, scale: 0.9 }}
      transition={{ duration: 0.3, type: "spring", stiffness: 300, damping: 25 }}
      whileHover={{ scale: 1.02 }}
      className={`relative group flex items-center justify-between p-3 cursor-pointer transition-all duration-300 rounded-lg overflow-hidden border ${
        isActive 
          ? 'bg-zinc-800/80 shadow-[inset_0_0_12px_rgba(255,255,255,0.05)] border-white/20' 
          : 'bg-transparent hover:bg-white/5 border-transparent hover:border-white/10'
      }`}
      onClick={() => onClick(id)}
    >
      <div className="flex items-center gap-3 overflow-hidden z-10 relative">
        <MessageSquare className={`w-4 h-4 shrink-0 transition-colors ${isActive ? 'text-zinc-200' : 'text-zinc-500 group-hover:text-zinc-400'}`} />
        <span className={`text-[0.95rem] font-medium tracking-wide truncate transition-colors ${isActive ? 'text-zinc-100' : 'text-zinc-400 group-hover:text-zinc-300'}`}>
          {title}
        </span>
      </div>
      
      <button
        onClick={(e) => {
          e.stopPropagation();
          onDelete(id);
        }}
        className={`z-10 relative opacity-0 group-hover:opacity-100 p-1.5 text-zinc-500 hover:text-red-400 hover:bg-white/10 transition-all rounded-md ${
          isActive ? 'opacity-100' : ''
        }`}
        title="Delete Chat"
      >
        <Trash2 className="w-[1.1rem] h-[1.1rem]" />
      </button>

      {/* Hover Gradient Overlay */}
      <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/5 to-white/0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />
    </motion.div>
  );
};
