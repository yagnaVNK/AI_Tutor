import React, { useState, useEffect } from 'react';

export default function SettingsModal({
  visible,
  onClose,
  initialPrompt,
  onSave,
}) {
  const [prompt, setPrompt] = useState(initialPrompt || '');

  useEffect(() => {
    setPrompt(initialPrompt || '');
  }, [initialPrompt, visible]);

  if (!visible) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="w-full max-w-lg bg-slate-900 border border-slate-800 rounded-2xl shadow-xl overflow-hidden animate-in zoom-in-95 duration-200">
        <div className="p-6">
          <h2 className="text-xl font-bold text-slate-100 mb-2">Custom system prompt</h2>
          <p className="text-sm text-slate-400 mb-6">
            This overrides the default behavior of the assistant for every
            new conversation. Per-conversation overrides take precedence.
          </p>
          
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="e.g. You are a strict math tutor that explains step by step..."
            className="w-full h-40 bg-slate-950 text-slate-200 border border-slate-800 rounded-xl p-4 text-sm placeholder-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all resize-none"
          />
        </div>
        
        <div className="flex items-center justify-end gap-3 px-6 py-4 bg-slate-950 border-t border-slate-800">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={() => onSave(prompt.trim())}
            className="px-4 py-2 text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 rounded-lg transition-colors shadow-sm"
          >
            Save Changes
          </button>
        </div>
      </div>
    </div>
  );
}
