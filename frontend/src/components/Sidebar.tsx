import React from 'react';
import { Branding } from '../sub-components/Branding';
import { ActionButton } from '../sub-components/ActionButton';
import { type ChatType, logout } from '../services/api-service';
import { useNavigate } from 'react-router-dom';
import { Plus, LogOut } from 'lucide-react';
import { ListButton } from '../sub-components/ListButton';
import { motion } from 'framer-motion';

interface SidebarProps {
  chats: ChatType[];
  activeChat: number | null;
  onSelectChat: (id: number) => void;
  onDeleteChat: (id: number) => void;
  onNewChat: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ chats, activeChat, onSelectChat, onDeleteChat, onNewChat }) => {
  const navigate = useNavigate();

  const handleLogout = async () => {
    try {
      await logout();
      navigate('/');
    } catch (e) {
      console.error("Logout failed", e);
    }
  };

  return (
    <div className="w-80 h-full bg-sidebar-bg border-r border-white/5 flex flex-col pt-4 pb-6 shadow-[4px_0_24px_rgba(0,0,0,0.5)] z-10">
      <div className="px-4 mb-6">
        <Branding />
      </div>

      <div className="px-6 mb-6">
        <ActionButton 
          label="New Chat" 
          icon={<Plus className="w-4 h-4" />} 
          onClick={onNewChat} 
          fullWidth 
        />
      </div>

      <div className="flex-1 overflow-y-auto w-full custom-scrollbar">
        <div className="text-xs font-semibold text-slate-500 uppercase tracking-widest pl-6 mb-3">
          Conversations
        </div>
        <div className="flex flex-col h-full w-full">
          {chats.length > 0 ? (
            <motion.div 
              initial="hidden"
              animate="visible"
              variants={{
                visible: { transition: { staggerChildren: 0.1 } }
              }}
              className="flex flex-col gap-1 px-4"
            >
              {chats.map((chat) => (
                <ListButton 
                  key={chat.id}
                  id={chat.id}
                  title={chat.title}
                  isActive={activeChat === chat.id}
                  onClick={onSelectChat}
                  onDelete={onDeleteChat}
                />
              ))}
            </motion.div>
          ) : (
             <div className="text-slate-500 text-sm text-center mt-4 italic">
               No chats yet.
             </div>
          )}
        </div>
      </div>

      <div className="px-6 mt-6 border-t border-white/5 pt-6">
        <ActionButton 
          variant="danger"
          label="Disconnect" 
          icon={<LogOut className="w-4 h-4" />} 
          onClick={handleLogout} 
          fullWidth 
        />
      </div>
    </div>
  );
};
