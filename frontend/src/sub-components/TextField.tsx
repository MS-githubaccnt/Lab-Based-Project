import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Send } from 'lucide-react';

interface TextFieldProps {
  onSend: (text: string) => void;
  isLoading: boolean;
}

export const TextField: React.FC<TextFieldProps> = ({ onSend, isLoading }) => {
  const [text, setText] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (text.trim() && !isLoading) {
      onSend(text);
      setText('');
    }
  };

  return (
    <motion.form 
      initial={{ y: 20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="relative flex items-center w-full max-w-4xl mx-auto p-2  bg-white/5 border-none shadow-2xl backdrop-blur-xl focus-within:ring-1 focus-within:ring-zinc-500/50 transition-all duration-300"
      onSubmit={handleSubmit}
    >
      <input
        type="text"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type a message..."
        disabled={isLoading}
        className="flex-1 bg-transparent border-none outline-none px-4 py-3 text-white placeholder:text-slate-500 disabled:opacity-50"
      />
      <motion.button
        type="submit"
        disabled={isLoading || !text.trim()}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        className="p-3 ml-2  bg-zinc-800 hover:bg-zinc-700 text-white disabled:bg-white/10 disabled:text-zinc-500 transition-colors"
      >
        <Send className="w-5 h-5" />
      </motion.button>
    </motion.form>
  );
};
