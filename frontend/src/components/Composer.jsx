import React, { useState, useRef } from 'react';
import { Paperclip, Mic, Send, X } from 'lucide-react';

export default function Composer({
  onSend,
  onUploadFile,
  onMicDown,
  onMicUp,
  micActive,
  attachedFiles,
  onRemoveFile,
  disabled,
}) {
  const [draft, setDraft] = useState('');
  const fileInputRef = useRef(null);

  const send = () => {
    const text = draft.trim();
    if (!text) return;
    onSend(text);
    setDraft('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  const handleUploadClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      onUploadFile(file);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="bg-slate-900 border-t border-slate-800 p-4 shrink-0">
      <div className="max-w-3xl mx-auto">
        {/* Attached Files row */}
        {attachedFiles?.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-3">
            {attachedFiles.map((f) => (
              <div
                key={f.id}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 border border-slate-700 rounded-full text-xs text-slate-300 max-w-[200px]"
              >
                <span className="truncate">{f.original_name}</span>
                <button
                  onClick={() => onRemoveFile(f.id)}
                  className="p-0.5 hover:bg-slate-700 rounded-full text-slate-400 hover:text-slate-200 transition-colors"
                >
                  <X size={14} />
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="flex items-end gap-2">
          {/* File Input (Hidden) */}
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileChange} 
            className="hidden" 
          />
          <button
            onClick={handleUploadClick}
            disabled={disabled}
            className="p-3 text-slate-400 hover:text-slate-200 bg-slate-800 hover:bg-slate-700 rounded-xl transition-colors disabled:opacity-50"
            title="Attach file"
          >
            <Paperclip size={20} />
          </button>

          <div className="flex-1 relative bg-slate-800 rounded-xl border border-slate-700 focus-within:border-indigo-500/50 focus-within:ring-1 focus-within:ring-indigo-500/50 transition-all">
            <textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your message..."
              disabled={disabled}
              className="w-full bg-transparent text-slate-100 placeholder-slate-500 px-4 py-3 min-h-[44px] max-h-[160px] outline-none resize-none disabled:opacity-50"
              rows={1}
              style={{
                height: draft ? 'auto' : '44px',
              }}
            />
          </div>

          <button
            onPointerDown={onMicDown}
            onPointerUp={onMicUp}
            onPointerLeave={onMicUp}
            disabled={disabled}
            className={`p-3 rounded-xl transition-all ${
              micActive
                ? 'bg-red-500 text-white animate-pulse'
                : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200 disabled:opacity-50'
            }`}
            title="Hold to speak"
          >
            <Mic size={20} />
          </button>

          <button
            onClick={send}
            disabled={!draft.trim() || disabled}
            className="p-3 bg-indigo-600 text-white rounded-xl hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:bg-slate-800 disabled:text-slate-500"
            title="Send message"
          >
            <Send size={20} className={draft.trim() ? "translate-x-0.5 -translate-y-0.5" : ""} />
          </button>
        </div>
      </div>
    </div>
  );
}
