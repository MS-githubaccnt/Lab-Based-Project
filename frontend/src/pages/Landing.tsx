import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { LoginForm } from '../components/LoginForm';
import { SignupForm } from '../components/SignupForm';
import { Branding } from '../sub-components/Branding';

export const Landing: React.FC = () => {
  const [isLogin, setIsLogin] = useState(true);

  return (
    <div className="min-h-screen w-full flex flex-col relative overflow-hidden bg-background">
      {/* Main Content */}
      <div className="relative z-10 flex-1 flex flex-col items-center justify-center p-6">
        <motion.div 
          initial={{ y: -50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8 }}
          className="mb-12"
        >
          <Branding />
          <p className="text-center text-slate-400 mt-4 max-w-md">
            
          </p>
        </motion.div>

        <div className="w-full max-w-md relative min-h-[500px]">
          <AnimatePresence mode="wait">
            {isLogin ? (
              <motion.div key="login" className="absolute w-full" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 20 }}>
                <LoginForm onToggle={() => setIsLogin(false)} />
              </motion.div>
            ) : (
              <motion.div key="signup" className="absolute w-full" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }}>
                <SignupForm onToggle={() => setIsLogin(true)} onSuccess={() => setIsLogin(true)} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
};
