import React, { useEffect, useState, useRef } from 'react';
import { Sidebar } from '../components/Sidebar';
import { ChatBox } from '../components/ChatBox';
import {
  getChats,
  createChat,
  deleteChat,
  updateChatTitle,
  startAnalysis,
  submitClarification,
  askFollowup,
  startPolling,
  clearSession,
  type ChatType,
  type ThoughtType,
  type AnalysisResult,
} from '../services/api-service';

// ---------------------------------------------------------------------------
// AnalysisMessage — replaces the old MessageType
// ---------------------------------------------------------------------------

export interface AnalysisMessage {
  /** task_id from the backend — unique per pipeline run / followup */
  id: string;
  /** Stable across clarification resumptions for the same analysis */
  sessionId: string;
  /** 'analysis' = CAD upload run; 'followup' = follow-up question */
  type: 'analysis' | 'followup';
  /** object_description for analysis, question text for followup */
  prompt: string;
  /** Original CAD filename — for display only, not stored after completion */
  fileName: string | null;
  /** Accumulated reasoning steps from all completed nodes */
  thoughts: ThoughtType[];
  /** Latest node status message e.g. "Parsing CAD file..." */
  progress: string | null;
  status: 'pending' | 'running' | 'clarification_needed' | 'complete' | 'failed';
  result: AnalysisResult | null;
  /** Populated when status='clarification_needed' */
  clarificationQuestion: string | null;
  /** Populated when status='failed' */
  error: string | null;
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

export const Dashboard: React.FC = () => {
  const [chats, setChats] = useState<ChatType[]>([]);
  const [activeChat, setActiveChat] = useState<number | null>(null);
  const [chatMessages, setChatMessages] = useState<Record<number, AnalysisMessage[]>>({});

  // cancelPolling is stored in a ref — it's mutable side-effect state, not UI state.
  // Keyed by chatId so we can cancel the active poll when switching chats.
  const cancelPollingRef = useRef<Record<number, (() => void) | undefined>>({});

  // Ref mirrors for stable closure access
  const activeChatRef = useRef(activeChat);
  useEffect(() => { activeChatRef.current = activeChat; }, [activeChat]);

  // ── Load chats on mount ──────────────────────────────────────────────────

  useEffect(() => {
    const loadChats = async () => {
      try {
        const data = await getChats();
        setChats(data);
        if (data.length > 0 && !activeChatRef.current) {
          setActiveChat(data[0].id);
        }
      } catch (e) {
        console.error('Error loading chats', e);
      }
    };
    loadChats();
  }, []);

  // ── Cancel polling on unmount ────────────────────────────────────────────

  useEffect(() => {
    return () => {
      Object.values(cancelPollingRef.current).forEach(cancel => cancel?.());
    };
  }, []);

  // ── Chat management (unchanged logic) ────────────────────────────────────

  const renameIfNew = async (id: number | null) => {
    if (!id) return;
    const chat = chats.find(c => c.id === id);
    if (chat?.title === 'New Chat') {
      try {
        const updated = await updateChatTitle(id);
        setChats(prev => prev.map(c => c.id === id ? { ...c, title: updated.title } : c));
      } catch (e) {
        console.error('Error renaming chat', e);
      }
    }
  };

  const handleNewChat = async () => {
    try {
      if (activeChat) {
        cancelPollingRef.current[activeChat]?.();
        await renameIfNew(activeChat);
      }
      const res = await createChat();
      const newChat: ChatType = { id: res.id, title: 'New Chat' };
      setChats(prev => [newChat, ...prev]);
      setActiveChat(newChat.id);
      setChatMessages(prev => ({ ...prev, [newChat.id]: [] }));
    } catch (e) {
      console.error('Error creating chat', e);
    }
  };

  const handleDeleteChat = async (id: number) => {
    try {
      // Cancel any running poll and clear the session before deleting
      cancelPollingRef.current[id]?.();
      delete cancelPollingRef.current[id];

      const lastMsg = (chatMessages[id] ?? []).findLast(m => m.type === 'analysis');
      if (lastMsg?.sessionId) {
        clearSession(lastMsg.sessionId).catch(() => {});
      }

      await deleteChat(id);
      const updated = chats.filter(c => c.id !== id);
      setChats(updated);
      if (activeChat === id) {
        setActiveChat(updated.length > 0 ? updated[0].id : null);
      }
    } catch (e) {
      console.error('Error deleting chat', e);
    }
  };

  const handleSelectChat = async (id: number) => {
    if (activeChat === id) return;
    // Cancel the poll for the chat we're leaving (it keeps running in the
    // background if we don't — the message will still update when we return)
    // Intentionally NOT cancelling: background updates are useful UX
    await renameIfNew(activeChat);
    setActiveChat(id);
  };

  // ── Core polling helper ───────────────────────────────────────────────────

  /**
   * Start polling a task_id and update the target message in state.
   * Uses the message's `id` field to find it for updates.
   * Returns the cancel function.
   */
  const attachPolling = (
    chatId: number,
    taskId: string,
    messageId: string,
  ): (() => void) => {
    const cancel = startPolling(taskId, {
      onUpdate: (taskStatus) => {
        setChatMessages(prev => {
          const msgs = prev[chatId] ?? [];
          return {
            ...prev,
            [chatId]: msgs.map(m =>
              m.id === messageId
                ? {
                    ...m,
                    status:   taskStatus.status,
                    progress: taskStatus.progress ?? m.progress,
                    thoughts: taskStatus.thoughts.length > 0
                      ? taskStatus.thoughts
                      : m.thoughts,
                    clarificationQuestion: taskStatus.clarification_question ?? null,
                    error: taskStatus.error ?? null,
                  }
                : m
            ),
          };
        });
      },

      onComplete: (taskStatus) => {
        setChatMessages(prev => {
          const msgs = prev[chatId] ?? [];
          return {
            ...prev,
            [chatId]: msgs.map(m =>
              m.id === messageId
                ? {
                    ...m,
                    status:   'complete',
                    progress: null,
                    thoughts: taskStatus.thoughts.length > 0
                      ? taskStatus.thoughts
                      : m.thoughts,
                    result: taskStatus.result ?? null,
                    error:  null,
                  }
                : m
            ),
          };
        });
        delete cancelPollingRef.current[chatId];
      },

      onClarification: (question, taskStatus) => {
        setChatMessages(prev => {
          const msgs = prev[chatId] ?? [];
          return {
            ...prev,
            [chatId]: msgs.map(m =>
              m.id === messageId
                ? {
                    ...m,
                    status:                'clarification_needed',
                    clarificationQuestion: question,
                    thoughts: taskStatus.thoughts.length > 0
                      ? taskStatus.thoughts
                      : m.thoughts,
                    progress: null,
                  }
                : m
            ),
          };
        });
        delete cancelPollingRef.current[chatId];
      },

      onError: (error, taskStatus) => {
        setChatMessages(prev => {
          const msgs = prev[chatId] ?? [];
          return {
            ...prev,
            [chatId]: msgs.map(m =>
              m.id === messageId
                ? {
                    ...m,
                    status:   'failed',
                    error,
                    thoughts: taskStatus.thoughts.length > 0
                      ? taskStatus.thoughts
                      : m.thoughts,
                    progress: null,
                  }
                : m
            ),
          };
        });
        delete cancelPollingRef.current[chatId];
      },
    });

    cancelPollingRef.current[chatId] = cancel;
    return cancel;
  };

  // ── Start new analysis ────────────────────────────────────────────────────

  const handleStartAnalysis = async (file: File, objectDescription: string, recyclabilityPriority: number) => {
    const chatId = activeChatRef.current;
    if (!chatId) return;

    // Cancel any previous in-flight poll for this chat
    cancelPollingRef.current[chatId]?.();

    // Optimistic message — shown immediately while the upload is in flight
    const tempId = `temp-${Date.now()}`;
    const optimistic: AnalysisMessage = {
      id:                    tempId,
      sessionId:             '',
      type:                  'analysis',
      prompt:                objectDescription,
      fileName:              file.name,
      thoughts:              [],
      progress:              'Uploading CAD file...',
      status:                'pending',
      result:                null,
      clarificationQuestion: null,
      error:                 null,
    };

    setChatMessages(prev => ({
      ...prev,
      [chatId]: [...(prev[chatId] ?? []), optimistic],
    }));

    try {
      const accepted = await startAnalysis(file, objectDescription, recyclabilityPriority);

      // Promote temp message to real one — swap id and set session_id
      setChatMessages(prev => {
        const msgs = prev[chatId] ?? [];
        return {
          ...prev,
          [chatId]: msgs.map(m =>
            m.id === tempId
              ? { ...m, id: accepted.task_id, sessionId: accepted.session_id, status: 'pending' }
              : m
          ),
        };
      });

      // Begin polling — message is found by its (now-real) task_id
      attachPolling(chatId, accepted.task_id, accepted.task_id);

    } catch (e) {
      // Remove optimistic message on error and surface it
      setChatMessages(prev => {
        const msgs = prev[chatId] ?? [];
        return {
          ...prev,
          [chatId]: msgs
            .filter(m => m.id !== tempId)
            .concat({
              ...optimistic,
              id:       `error-${Date.now()}`,
              status:   'failed',
              progress: null,
              error:    e instanceof Error ? e.message : 'Failed to start analysis.',
            }),
        };
      });
    }
  };

  // ── Submit clarification answer ───────────────────────────────────────────

  const handleSubmitClarification = async (sessionId: string, answer: string) => {
    const chatId = activeChatRef.current;
    if (!chatId) return;

    // Find the message that is waiting for clarification
    const targetMsg = (chatMessages[chatId] ?? [])
      .findLast(m => m.sessionId === sessionId && m.status === 'clarification_needed');
    if (!targetMsg) return;

    // Set it back to running while we resume
    setChatMessages(prev => ({
      ...prev,
      [chatId]: (prev[chatId] ?? []).map(m =>
        m.id === targetMsg.id
          ? { ...m, status: 'running', progress: 'Resuming analysis...', clarificationQuestion: null }
          : m
      ),
    }));

    try {
      const accepted = await submitClarification(sessionId, answer);
      // Update message id to the new task_id for the resumed run
      setChatMessages(prev => ({
        ...prev,
        [chatId]: (prev[chatId] ?? []).map(m =>
          m.id === targetMsg.id
            ? { ...m, id: accepted.task_id }
            : m
        ),
      }));
      attachPolling(chatId, accepted.task_id, accepted.task_id);
    } catch (e) {
      setChatMessages(prev => ({
        ...prev,
        [chatId]: (prev[chatId] ?? []).map(m =>
          m.id === targetMsg.id
            ? { ...m, status: 'failed', error: e instanceof Error ? e.message : 'Failed to resume.', progress: null }
            : m
        ),
      }));
    }
  };

  // ── Ask follow-up question ────────────────────────────────────────────────

  const handleAskFollowup = async (sessionId: string, question: string) => {
    const chatId = activeChatRef.current;
    if (!chatId) return;

    cancelPollingRef.current[chatId]?.();

    const tempId = `followup-temp-${Date.now()}`;
    const optimistic: AnalysisMessage = {
      id:                    tempId,
      sessionId,
      type:                  'followup',
      prompt:                question,
      fileName:              null,
      thoughts:              [],
      progress:              'Answering your question...',
      status:                'pending',
      result:                null,
      clarificationQuestion: null,
      error:                 null,
    };

    setChatMessages(prev => ({
      ...prev,
      [chatId]: [...(prev[chatId] ?? []), optimistic],
    }));

    try {
      const accepted = await askFollowup(sessionId, question);

      setChatMessages(prev => ({
        ...prev,
        [chatId]: (prev[chatId] ?? []).map(m =>
          m.id === tempId
            ? { ...m, id: accepted.task_id, status: 'pending' }
            : m
        ),
      }));

      attachPolling(chatId, accepted.task_id, accepted.task_id);

    } catch (e) {
      setChatMessages(prev => ({
        ...prev,
        [chatId]: (prev[chatId] ?? []).map(m =>
          m.id === tempId
            ? { ...m, status: 'failed', error: e instanceof Error ? e.message : 'Failed to send question.', progress: null }
            : m
        ),
      }));
    }
  };

  // ── Render ────────────────────────────────────────────────────────────────

  const activeMessages = activeChat ? (chatMessages[activeChat] ?? []) : [];
  const lastMsg = activeMessages.at(-1);

  // Derive the active session_id for the current chat:
  // the session_id of the most recent analysis-type message
  const activeSessionId =
    [...activeMessages].reverse().find(m => m.type === 'analysis')?.sessionId ?? null;

  // The input is disabled while any message in this chat is actively running
  const isLoading =
    lastMsg?.status === 'pending' || lastMsg?.status === 'running';

  return (
    <div className="flex h-screen w-full bg-background overflow-hidden relative">
      <Sidebar
        chats={chats}
        activeChat={activeChat}
        onSelectChat={handleSelectChat}
        onDeleteChat={handleDeleteChat}
        onNewChat={handleNewChat}
      />

      {activeChat ? (
        <ChatBox
          messages={activeMessages}
          sessionId={activeSessionId}
          isLoading={isLoading}
          onStartAnalysis={handleStartAnalysis}
          onSubmitClarification={handleSubmitClarification}
          onAskFollowup={handleAskFollowup}
        />
      ) : (
        <div className="flex-1 flex flex-col items-center justify-center text-slate-500">
          <p>No active chat. Create a new one to begin.</p>
        </div>
      )}
    </div>
  );
};