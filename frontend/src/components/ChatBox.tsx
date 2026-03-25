import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ResponseBox } from '../sub-components/ResponseBox';
import { ThoughtBox } from '../sub-components/ThoughtBox';
import { TextField } from '../sub-components/TextField';
import LightRays from '../sub-components/react-bits/LightRays/LightRays';
import { type AnalysisMessage } from '../pages/Dashboard';

// ---------------------------------------------------------------------------
// Input mode — derived from the messages array
// ---------------------------------------------------------------------------

type InputMode =
  | 'analysis'       // no messages yet, or last is complete/failed — show file upload
  | 'followup'       // last analysis message is complete — show text-only input
  | 'clarification'  // last message is waiting for clarification — show answer input
  | 'disabled';      // pipeline is running — input locked

function deriveInputMode(messages: AnalysisMessage[]): InputMode {
  const last = messages.at(-1);
  if (!last) return 'analysis';
  if (last.status === 'pending' || last.status === 'running') return 'disabled';
  if (last.status === 'clarification_needed') return 'clarification';
  if (last.status === 'complete') return 'followup';
  // failed or any other terminal state — allow starting a fresh analysis
  return 'analysis';
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface ChatBoxProps {
  messages: AnalysisMessage[];
  /** The session_id of the most recent analysis in this chat. Null if no analysis yet. */
  sessionId: string | null;
  isLoading: boolean;
  onStartAnalysis: (file: File, objectDescription: string, recyclabilityPriority: number) => void;
  onSubmitClarification: (sessionId: string, answer: string) => void;
  onAskFollowup: (sessionId: string, question: string) => void;
}

// ---------------------------------------------------------------------------
// ChatBox
// ---------------------------------------------------------------------------

export const ChatBox: React.FC<ChatBoxProps> = ({
  messages,
  sessionId,
  isLoading,
  onStartAnalysis,
  onSubmitClarification,
  onAskFollowup,
}) => {
  const bottomRef  = useRef<HTMLDivElement>(null);
  const inputMode  = deriveInputMode(messages);

  // Local state for the analysis input form
  const [selectedFile, setSelectedFile]   = useState<File | null>(null);
  const [description, setDescription]     = useState('');
  const [ecoWeight, setEcoWeight]         = useState(0.7);
  const [clarifyAnswer, setClarifyAnswer] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom when messages or thoughts update
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages]);

  // ── Submit handlers ────────────────────────────────────────────────────

  const handleAnalysisSubmit = () => {
    if (!selectedFile || !description.trim()) return;
    onStartAnalysis(selectedFile, description.trim(), ecoWeight);
    // Reset form
    setSelectedFile(null);
    setDescription('');
    setEcoWeight(0.7);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleClarifySubmit = () => {
    if (!clarifyAnswer.trim() || !sessionId) return;
    onSubmitClarification(sessionId, clarifyAnswer.trim());
    setClarifyAnswer('');
  };

  const handleFollowupSend = (text: string) => {
    if (!sessionId) return;
    onAskFollowup(sessionId, text);
  };

  // ── Render helpers ─────────────────────────────────────────────────────

  const renderMessage = (msg: AnalysisMessage) => {
    const isProcessing = msg.status === 'pending' || msg.status === 'running';
    const explanation  = msg.result?.report?.explanation?.text ?? null;
    const followupAns  = msg.result?.followup_answer ?? null;

    return (
      <div key={msg.id} className="flex flex-col w-full gap-2">

        {/* User prompt bubble */}
        <div className="flex justify-end">
          <div className="max-w-[70%] rounded-2xl bg-white/10 border border-white/10 px-4 py-3 text-sm text-white">
            {msg.fileName && (
              <p className="text-xs text-slate-400 mb-1 font-mono">
                📎 {msg.fileName}
              </p>
            )}
            <p>{msg.prompt}</p>
          </div>
        </div>

        {/* Thoughts — visible while running and after completion */}
        <AnimatePresence>
          {msg.thoughts.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
            >
              <ThoughtBox thoughts={msg.thoughts} isStreaming={isProcessing} />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Progress pulse — shown while running with no thoughts yet */}
        {isProcessing && msg.thoughts.length === 0 && (
          <div className="flex items-center gap-2 text-slate-400 text-sm px-2">
            <span className="inline-block w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
            {msg.progress ?? 'Processing...'}
          </div>
        )}

        {/* Progress message while running (thoughts present) */}
        {isProcessing && msg.thoughts.length > 0 && msg.progress && (
          <p className="text-xs text-slate-500 px-2">{msg.progress}</p>
        )}

        {/* Clarification question */}
        {msg.status === 'clarification_needed' && msg.clarificationQuestion && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            className="self-start max-w-[85%] rounded-2xl bg-amber-500/10 border border-amber-500/20 px-4 py-3 text-sm text-amber-200"
          >
            <p className="text-xs font-semibold text-amber-400 mb-1 uppercase tracking-wider">
              Clarification needed
            </p>
            <p>{msg.clarificationQuestion}</p>
          </motion.div>
        )}

        {/* Final explanation / followup answer */}
        {msg.status === 'complete' && (explanation || followupAns) && (
          <motion.div
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
          >
            <ResponseBox
              prompt={msg.prompt}
              response={explanation ?? followupAns}
              // report={msg.result?.report ?? null}
              // warnings={msg.result?.warnings ?? []}
            />
          </motion.div>
        )}

        {/* Warnings */}
        {msg.status === 'complete' && (msg.result?.warnings?.length ?? 0) > 0 && (
          <div className="flex flex-col gap-1 px-2">
            {msg.result!.warnings.map((w, i) => (
              <p key={i} className="text-xs text-amber-400/80">⚠ {w}</p>
            ))}
          </div>
        )}

        {/* Error state */}
        {msg.status === 'failed' && msg.error && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="self-start max-w-[85%] rounded-2xl bg-red-500/10 border border-red-500/20 px-4 py-3 text-sm text-red-300"
          >
            <p className="text-xs font-semibold text-red-400 mb-1 uppercase tracking-wider">
              Analysis failed
            </p>
            <p>{msg.error}</p>
          </motion.div>
        )}
      </div>
    );
  };

  // ── Input area — switches based on inputMode ───────────────────────────

  const renderInput = () => {
    if (inputMode === 'disabled') {
      return (
        <div className="flex items-center gap-2 text-slate-500 text-sm px-2 py-3">
          <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
          Pipeline running...
        </div>
      );
    }

    if (inputMode === 'clarification') {
      return (
        <div className="flex flex-col gap-2">
          <p className="text-xs text-amber-400/80 px-1">Answer the question above to continue the analysis</p>
          <div className="flex gap-2">
            <input
              className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-amber-400/40 transition-colors"
              placeholder="Your answer..."
              value={clarifyAnswer}
              onChange={e => setClarifyAnswer(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleClarifySubmit()}
            />
            <button
              onClick={handleClarifySubmit}
              disabled={!clarifyAnswer.trim()}
              className="px-5 py-3 rounded-xl bg-amber-500/20 border border-amber-500/30 text-amber-300 text-sm font-medium hover:bg-amber-500/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              Submit
            </button>
          </div>
        </div>
      );
    }

    if (inputMode === 'followup') {
      return (
        <div className="flex flex-col gap-1">
          <p className="text-xs text-slate-500 px-1">Ask a follow-up question about this recommendation</p>
          <TextField
            onSend={handleFollowupSend}
            isLoading={false}
            //placeholder="e.g. Why was hemp composite rejected?"
          />
        </div>
      );
    }

    // inputMode === 'analysis' — file upload + description
    return (
      <div className="flex flex-col gap-3">
        {/* File drop area */}
        <div
          onClick={() => fileInputRef.current?.click()}
          className={`flex items-center gap-3 px-4 py-3 rounded-xl border-2 border-dashed cursor-pointer transition-colors text-sm
            ${selectedFile
              ? 'border-green-500/40 bg-green-500/5 text-green-300'
              : 'border-white/10 bg-white/5 text-slate-400 hover:border-white/20 hover:bg-white/8'
            }`}
        >
          <span className="text-lg">{selectedFile ? '✅' : '📁'}</span>
          <span>
            {selectedFile
              ? `${selectedFile.name} (${(selectedFile.size / 1024).toFixed(0)} KB)`
              : 'Click to upload CAD file (.step, .stp, .stl)'}
          </span>
          {selectedFile && (
            <button
              onClick={e => { e.stopPropagation(); setSelectedFile(null); if (fileInputRef.current) fileInputRef.current.value = ''; }}
              className="ml-auto text-slate-500 hover:text-red-400 transition-colors text-xs"
            >
              ✕ Remove
            </button>
          )}
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept=".step,.stp,.stl"
          className="hidden"
          onChange={e => setSelectedFile(e.target.files?.[0] ?? null)}
        />

        {/* Description input */}
        <input
          className="bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-400/40 transition-colors"
          placeholder="Describe the part, e.g. bicycle crank arm"
          value={description}
          onChange={e => setDescription(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleAnalysisSubmit()}
        />

        {/* Eco priority + submit row */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 flex-1 text-xs text-slate-400">
            <span className="whitespace-nowrap">Eco priority</span>
            <input
              type="range"
              min={0} max={1} step={0.05}
              value={ecoWeight}
              onChange={e => setEcoWeight(Number(e.target.value))}
              className="flex-1 accent-green-400"
            />
            <span className="w-8 text-right">{ecoWeight.toFixed(2)}</span>
          </div>

          <button
            onClick={handleAnalysisSubmit}
            disabled={!selectedFile || !description.trim()}
            className="px-5 py-2.5 rounded-xl bg-blue-500/20 border border-blue-500/30 text-blue-300 text-sm font-medium hover:bg-blue-500/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
          >
            Analyse →
          </button>
        </div>
      </div>
    );
  };

  // ── Main render ─────────────────────────────────────────────────────────

  return (
    <div className="flex-1 h-full flex flex-col relative overflow-hidden bg-background">

      {/* Background */}
      <div className="absolute inset-0 pointer-events-none opacity-40 z-0">
        <LightRays />
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-8 py-10 custom-scrollbar z-10">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-slate-500">
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.5 }}
              className="w-24 h-24 mb-6 bg-white/5 border border-white/10 flex items-center justify-center shadow-2xl"
            >
              <div className="w-12 h-12 bg-blue-500/20 blur-xl animate-pulse" />
              <div className="absolute text-3xl">🌿</div>
            </motion.div>
            <h3 className="text-xl font-medium text-white mb-2">
              Eco-material analysis
            </h3>
            <p className="text-sm font-light text-center max-w-xs">
              Upload a CAD file and describe your part to get an eco-friendly material recommendation.
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-6">
            {messages.map(renderMessage)}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Input area */}
      <div className="p-6 bg-gradient-to-t from-background via-background to-transparent z-10 pb-8">
        {renderInput()}
      </div>

    </div>
  );
};