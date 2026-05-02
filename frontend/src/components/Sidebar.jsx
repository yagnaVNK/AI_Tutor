import React, { useState } from 'react';
import { LogOut, Settings, Plus, MessageSquare } from 'lucide-react';

export default function Sidebar({
  user,
  conversations,
  activeId,
  onSelect,
  onNew,
  onSignOut,
  onOpenSettings,
}) {
  return (
    <div className="w-72 bg-slate-950 flex flex-col border-r border-slate-800">
      <div className="p-4">
        <h1 className="text-xl font-bold text-slate-100 mb-4 px-2">AI Tutor</h1>
        <button
          onClick={onNew}
          className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-2.5 px-4 rounded-lg flex items-center justify-center gap-2 transition-colors"
        >
          <Plus size={18} />
          <span>New conversation</span>
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-4 pb-4 space-y-1">
        {conversations.map((c) => {
          const active = c.id === activeId;
          return (
            <button
              key={c.id}
              onClick={() => onSelect(c)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-colors ${
                active ? 'bg-slate-800 text-slate-100 font-medium' : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'
              }`}
            >
              <MessageSquare size={16} />
              <span className="truncate flex-1">{c.title || 'Untitled'}</span>
            </button>
          );
        })}
        {conversations.length === 0 && (
          <p className="text-slate-500 italic px-2 py-4 text-sm text-center">No conversations yet.</p>
        )}
      </div>

      <div className="p-4 border-t border-slate-800">
        <p className="text-slate-400 text-sm truncate mb-3 px-2">
          {user?.email || user?.displayName}
        </p>
        <div className="flex gap-2">
          <button
            onClick={onOpenSettings}
            className="flex-1 flex items-center justify-center gap-2 bg-slate-800 hover:bg-slate-700 text-slate-300 py-2 rounded-md text-sm transition-colors"
          >
            <Settings size={16} />
            <span>Settings</span>
          </button>
          <button
            onClick={onSignOut}
            className="flex-1 flex items-center justify-center gap-2 bg-slate-800 hover:bg-slate-700 text-slate-300 py-2 rounded-md text-sm transition-colors"
          >
            <LogOut size={16} />
            <span>Sign out</span>
          </button>
        </div>
      </div>
    </div>
  );
}
