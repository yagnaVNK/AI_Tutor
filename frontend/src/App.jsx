import React, { useEffect, useState } from 'react';
import LoginScreen from './screens/LoginScreen';
import ChatScreen from './screens/ChatScreen';
import { subscribeToAuth } from './services/firebase';

export default function App() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const unsub = subscribeToAuth((u) => {
      setUser(u);
      setLoading(false);
    });
    return unsub;
  }, []);

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-slate-950">
        <div className="w-10 h-10 border-4 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <div className="h-screen bg-slate-900">
      {user ? <ChatScreen user={user} /> : <LoginScreen />}
    </div>
  );
}
