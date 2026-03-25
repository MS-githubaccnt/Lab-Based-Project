import React from 'react';
import ShinyText from './react-bits/ShinyText/ShinyText';

export const Branding: React.FC = () => {
  return (
    <div className="flex items-center justify-center p-4">
      <div className="font-oswald text-[1.5rem] font-bold text-white tracking-widest drop-shadow-[0_0_8px_rgba(255,255,255,0.3)]">
        <ShinyText text="LAB BASED PROJECT" disabled={false} speed={3} className="" />
      </div>
    </div>
  );
};
