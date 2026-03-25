import React from 'react';
import ScrollReveal from './react-bits/ScrollReveal/ScrollReveal';
import ShinyText from './react-bits/ShinyText/ShinyText';

interface ThoughtType {
  node: string;
  type: string;
  text: string;
}

interface ThoughtBoxProps {
  thoughts: ThoughtType[];
  isStreaming?: boolean;
}

export const ThoughtBox: React.FC<ThoughtBoxProps> = ({ thoughts }) => {
  if (thoughts.length === 0) return null;

  return (
    <div className="w-full max-w-4xl mx-auto my-3 px-6 py-5 bg-gradient-to-r from-zinc-900/50 to-transparent rounded-xl border border-zinc-800/50 italic text-slate-300 font-light text-[0.9rem] shadow-[0_4px_24px_rgba(0,0,0,0.2)]">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-1.5 h-1.5 rounded-full bg-blue-400 shadow-[0_0_8px_rgba(96,165,250,0.8)] animate-pulse" />
        <div className="text-xs uppercase tracking-widest font-medium">
          <ShinyText text="AI Thinking Context" disabled={false} speed={2} className="text-blue-200/80" />
        </div>
      </div>
      <div className="flex flex-col gap-2 pl-4 border-l-2 border-blue-500/20">
        {thoughts.map((thought, idx) => (
          <ScrollReveal
            key={idx}
            baseOpacity={0}
            baseRotation={0}
            blurStrength={4}
            containerClassName="my-1"
            textClassName="leading-relaxed"
          >
            {thought.text}
          </ScrollReveal>
        ))}
      </div>
    </div>
  );
};
