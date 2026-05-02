import React, { useEffect, useRef } from 'react';

export default function MessageList({ messages, streamingText }) {
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, streamingText]);

  return (
    <div className="flex-1 overflow-y-auto p-4 sm:p-8 bg-slate-900" ref={scrollRef}>
      <div className="max-w-3xl mx-auto space-y-6 pb-24">
        {messages.map((m) => (
          <Bubble key={m.id} message={m} />
        ))}
        {streamingText ? (
          <Bubble
            message={{
              id: 'streaming',
              role: 'assistant',
              content: streamingText + ' ▍',
            }}
          />
        ) : null}
        {messages.length === 0 && !streamingText && (
          <div className="flex justify-center items-center h-48">
            <p className="text-slate-500 text-sm">
              Type a message or hold the mic to start talking.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function Bubble({ message }) {
  const isUser = message.role === 'user';
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[85%] sm:max-w-[75%] px-5 py-3.5 rounded-2xl text-[15px] leading-relaxed ${
          isUser
            ? 'bg-indigo-600 text-white rounded-br-sm'
            : 'bg-slate-800 text-slate-200 rounded-bl-sm shadow-sm border border-slate-700/50'
        }`}
      >
        <p className="whitespace-pre-wrap break-words">{message.content}</p>
      </div>
    </div>
  );
}
