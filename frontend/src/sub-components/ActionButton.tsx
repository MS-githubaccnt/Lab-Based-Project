import React from 'react';
import { motion } from 'framer-motion';

interface ActionButtonProps {
  label: string;
  onClick: () => void;
  icon?: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'danger';
  fullWidth?: boolean;
}

export const ActionButton: React.FC<ActionButtonProps> = ({ 
  label, 
  onClick, 
  icon, 
  variant = 'primary',
  fullWidth = false
}) => {
  const baseStyles = "flex items-center justify-center gap-2 px-4 py-2  font-medium transition-all duration-300 transform active:scale-95";
  const variants = {
    primary: "bg-zinc-800 hover:bg-zinc-700 text-white shadow-[0_0_10px_rgba(255,255,255,0.1)]",
    secondary: "bg-white/5 hover:bg-white/10 text-white",
    danger: "bg-zinc-800/50 hover:bg-zinc-700/50 text-zinc-300"
  };

  return (
    <motion.button
      whileHover={{ y: -2 }}
      className={`${baseStyles} ${variants[variant]} ${fullWidth ? 'w-full' : ''}`}
      onClick={onClick}
    >
      {icon}
      <span>{label}</span>
    </motion.button>
  );
};
