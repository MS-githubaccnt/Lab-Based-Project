import React from 'react';
import { motion } from 'framer-motion';
import { User, Sparkles } from 'lucide-react';
import ScrollReveal from './react-bits/ScrollReveal/ScrollReveal';

interface ResponseBoxProps {
  prompt: string;
  response?: string | null;
}

const formatResponse = (res: unknown): string => {
  if (!res) return "";
  if (typeof res === "string") return res;

  try {
    return JSON.stringify(res, null, 2);
  } catch {
    return "Unable to render response.";
  }
};

export const ResponseBox: React.FC<ResponseBoxProps> = ({ prompt, response }) => {
  const formatted = formatResponse(response);

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full max-w-4xl mx-auto my-6 flex flex-col gap-6"
    >
      {/* Prompt */}
      <div className="self-end max-w-[80%]">
        <div className="flex items-start gap-4 flex-row-reverse">
          <div className="w-8 h-8 rounded-full bg-white/10 flex items-center justify-center">
            <User className="w-4 h-4 text-white" />
          </div>
          <div className="px-6 py-3 rounded-2xl bg-blue-900/30 text-white border border-blue-500/20">
            {prompt}
          </div>
        </div>
      </div>

      {/* Response */}
      {response != null && (
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="self-start max-w-[90%]"
        >
          <div className="flex items-start gap-4">
            <div className="w-8 h-8 rounded-full bg-zinc-800 flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-white" />
            </div>

            <div className="px-6 py-4 rounded-2xl bg-zinc-900 text-zinc-100 border border-white/10">
              <ScrollReveal baseOpacity={0} blurStrength={3}>
                {formatted}
              </ScrollReveal>
            </div>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
};