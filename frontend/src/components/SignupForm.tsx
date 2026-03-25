import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { ActionButton } from '../sub-components/ActionButton';
import { auth, googleProvider } from '../services/firebase';
import { createUserWithEmailAndPassword, signInWithPopup } from 'firebase/auth';

export const SignupForm: React.FC<{ onToggle: () => void, onSuccess: () => void }> = ({ onToggle, onSuccess }) => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const getStrength = (pass: string) => {
    let score = 0;
    if (pass.length > 5) score++;
    if (pass.length > 8) score++;
    if (/[A-Z]/.test(pass)) score++;
    if (/[0-9]/.test(pass)) score++;
    if (/[^a-zA-Z0-9]/.test(pass)) score++;
    return score;
  };

  const strength = getStrength(password);
  const strengthColors = ['bg-white/10', 'bg-white/30', 'bg-white/50', 'bg-white/70', 'bg-white/90', 'bg-white'];

  const handleGoogleSignUp = async () => {
    setError('');
    setIsLoading(true);
    try {
      await signInWithPopup(auth, googleProvider);
      setTimeout(() => onSuccess(), 500);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message || "Google Sign-In failed.");
      } else {
        setError("Google Sign-In failed.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    
    if (password !== confirm) {
      setError("Passwords do not match");
      return;
    }
    
    setIsLoading(true);
    try {
      await createUserWithEmailAndPassword(auth, email, password);
      // Wait a moment for Firebase auth state to update before proceeding
      setTimeout(() => onSuccess(), 500);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message || "Registration failed. Please try again.");
      } else {
        setError("Registration failed. Please try again.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      className="w-full max-w-md p-8  glass-panel text-white"
    >
      <h2 className="text-3xl font-bold mb-6 text-center">Create Account</h2>
      {error && <div className="p-3 mb-4 bg-zinc-800/50 text-zinc-300 text-sm text-center border-none">{error}</div>}
      
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <div>
          <label className="block text-sm font-medium text-slate-400 mb-1">Name</label>
          <input required type="text" value={name} onChange={e => setName(e.target.value)}
            className="w-full bg-white/5 border-none px-4 py-3 placeholder:text-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-400 transition-all"
            placeholder="John Doe" />
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-400 mb-1">Email</label>
          <input required type="email" value={email} onChange={e => setEmail(e.target.value)}
            className="w-full bg-white/5 border-none px-4 py-3 placeholder:text-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-400 transition-all"
            placeholder="contact@example.com" />
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-400 mb-1">Password</label>
          <input required type="password" value={password} onChange={e => setPassword(e.target.value)}
            className="w-full bg-white/5 border-none px-4 py-3 placeholder:text-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-400 transition-all"
            placeholder="••••••••" />
          
          {/* Strength Indicator */}
          {password && (
            <div className="flex gap-1 mt-2 h-1.5 w-full">
              {[...Array(5)].map((_, i) => (
                <div key={i} className={`flex-1  transition-all duration-300 ${i < strength ? strengthColors[strength] : 'bg-white/10'}`} />
              ))}
            </div>
          )}
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-400 mb-1">Confirm Password</label>
          <input required type="password" value={confirm} onChange={e => setConfirm(e.target.value)}
            className="w-full bg-white/5 border-none px-4 py-3 placeholder:text-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-400 transition-all"
            placeholder="••••••••" />
        </div>
        
        <div className="mt-4 flex flex-col gap-3">
          <ActionButton label={isLoading ? "Creating..." : "Sign Up"} onClick={() => {}} fullWidth />
          
          <div className="relative flex items-center py-2">
            <div className="flex-grow border-t border-zinc-700"></div>
            <span className="flex-shrink-0 mx-4 text-zinc-500 text-sm">or</span>
            <div className="flex-grow border-t border-zinc-700"></div>
          </div>

          <button 
            type="button" 
            onClick={handleGoogleSignUp}
            disabled={isLoading}
            className="w-full flex items-center justify-center gap-2 bg-white/5 hover:bg-white/10 text-white font-medium py-3 px-4 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <svg className="w-5 h-5" viewBox="0 0 24 24">
              <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
              <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
              <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
              <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
            </svg>
            Sign up with Google
          </button>
        </div>
      </form>
      
      <p className="mt-6 text-center text-sm text-slate-400">
        Already have an account?{' '}
        <button onClick={onToggle} className="text-zinc-400 hover:text-zinc-300 transition-colors">Login</button>
      </p>
    </motion.div>
  );
};
