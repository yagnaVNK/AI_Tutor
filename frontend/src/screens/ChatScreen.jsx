import React, { useCallback, useEffect, useRef, useState } from 'react';
import Composer from '../components/Composer';
import MessageList from '../components/MessageList';
import SettingsModal from '../components/SettingsModal';
import Sidebar from '../components/Sidebar';
import { api } from '../services/api';
import { signOut } from '../services/firebase';
import { VoiceClient } from '../services/voiceClient';
import { Menu, X } from 'lucide-react';

export default function ChatScreen({ user }) {
  const [conversations, setConversations] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [streaming, setStreaming] = useState('');
  const [attachedFiles, setAttachedFiles] = useState([]);
  const [profile, setProfile] = useState(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [micActive, setMicActive] = useState(false);
  const [busy, setBusy] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  
  const voiceRef = useRef(null);

  // Load profile + conversation list on mount.
  useEffect(() => {
    (async () => {
      try {
        const [me, convs] = await Promise.all([
          api.getMe(),
          api.listConversations(),
        ]);
        setProfile(me);
        setConversations(convs);
        if (convs.length > 0) {
          await openConversation(convs[0]);
        }
      } catch (e) {
        console.warn('Initial load failed', e);
      }
    })();
    return () => {
      voiceRef.current?.disconnect();
    };
  }, []);

  const ensureVoice = useCallback(async () => {
    if (voiceRef.current && voiceRef.current.connected) return voiceRef.current;
    voiceRef.current?.disconnect();
    const client = new VoiceClient({
      conversationId: activeId,
      handlers: {
        ready: ({ conversation_id }) => {
          if (conversation_id && conversation_id !== activeId) {
            setActiveId(conversation_id);
            api.listConversations().then(setConversations).catch(() => {});
          }
        },
        transcript: (text) => {
          if (!text) return;
          setMessages((prev) => [
            ...prev,
            { id: `u-${Date.now()}`, role: 'user', content: text },
          ]);
        },
        assistantChunk: (chunk) => setStreaming((prev) => prev + chunk),
        assistantDone: (full) => {
          setMessages((prev) => [
            ...prev,
            { id: `a-${Date.now()}`, role: 'assistant', content: full },
          ]);
          setStreaming('');
          setBusy(false);
          api.listConversations().then(setConversations).catch(() => {});
        },
        interrupt: (payload) => {
          console.log("Interrupted by VAD");
          setBusy(false);
          setStreaming(''); // clear any pending streaming text on interrupt
        },
        error: (detail) => {
          console.warn('voice error', detail);
          setBusy(false);
          setStreaming('');
        },
        close: () => setBusy(false),
      },
    });
    await client.connect();
    voiceRef.current = client;
    return client;
  }, [activeId]);

  const openConversation = async (conv) => {
    setActiveId(conv.id);
    setStreaming('');
    setMessages([]);
    if (window.innerWidth < 768) setSidebarOpen(false);
    
    try {
      const detail = await api.getConversation(conv.id);
      setMessages(detail.messages || []);
    } catch (e) {
      console.warn('Load conversation failed', e);
    }
    voiceRef.current?.disconnect();
    voiceRef.current = null;
  };

  const newConversation = async () => {
    try {
      const conv = await api.createConversation({ title: 'New conversation' });
      setConversations((prev) => [conv, ...prev]);
      await openConversation(conv);
    } catch (e) {
      console.warn('Create conversation failed', e);
    }
  };

  const sendText = async (text) => {
    setBusy(true);
    setMessages((prev) => [
      ...prev,
      { id: `u-${Date.now()}`, role: 'user', content: text },
    ]);
    
    // Stop playback if we type something manually
    if (voiceRef.current) {
        voiceRef.current.stopPlayback();
    }
    
    try {
      const res = await api.sendChat({
        message: text,
        conversationId: activeId,
        fileIds: attachedFiles.map((f) => f.id),
      });
      if (!activeId) {
        setActiveId(res.conversation_id);
        const convs = await api.listConversations();
        setConversations(convs);
      }
      setMessages((prev) => [...prev, res.message]);
    } catch (e) {
      console.warn('Send failed', e);
    } finally {
      setBusy(false);
    }
  };

  const onMicDown = async () => {
    try {
      setBusy(true);
      setStreaming('');
      const client = await ensureVoice();
      await client.startRecording();
      setMicActive(true);
    } catch (e) {
      console.warn('mic start failed', e);
      setBusy(false);
    }
  };

  const onMicUp = async () => {
    if (!voiceRef.current) {
      setMicActive(false);
      setBusy(false);
      return;
    }
    setMicActive(false);
    await voiceRef.current.stopRecording();
  };

  const onUploadFile = async (file) => {
    if (!file) return;
    try {
      const uploaded = await api.uploadFile(file, activeId);
      setAttachedFiles((prev) => [...prev, uploaded]);
    } catch (err) {
      console.warn('Upload failed', err);
    }
  };

  const removeFile = (id) =>
    setAttachedFiles((prev) => prev.filter((f) => f.id !== id));

  const saveSystemPrompt = async (prompt) => {
    try {
      const updated = await api.updateMe({ custom_system_prompt: prompt });
      setProfile(updated);
      voiceRef.current?.setSystemPrompt(prompt);
    } catch (e) {
      console.warn('Save prompt failed', e);
    } finally {
      setSettingsOpen(false);
    }
  };

  return (
    <div className="flex h-screen w-full bg-slate-900 overflow-hidden relative">
      {/* Mobile header / sidebar toggle */}
      <div className="md:hidden absolute top-0 left-0 w-full p-4 flex items-center bg-slate-900/80 backdrop-blur border-b border-slate-800 z-20">
        <button 
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-lg"
        >
          {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
        <span className="font-semibold ml-3">AI Tutor</span>
      </div>

      {/* Sidebar overlay for mobile */}
      {sidebarOpen && (
        <div 
          className="md:hidden fixed inset-0 bg-black/50 z-30"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div className={`
        fixed md:static inset-y-0 left-0 z-40 transform transition-transform duration-300 ease-in-out flex
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
      `}>
        <Sidebar
          user={user}
          conversations={conversations}
          activeId={activeId}
          onSelect={openConversation}
          onNew={newConversation}
          onSignOut={() => signOut()}
          onOpenSettings={() => setSettingsOpen(true)}
        />
      </div>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0 pt-14 md:pt-0">
        <MessageList messages={messages} streamingText={streaming} />
        <Composer
          onSend={sendText}
          onUploadFile={onUploadFile}
          onMicDown={onMicDown}
          onMicUp={onMicUp}
          micActive={micActive}
          attachedFiles={attachedFiles}
          onRemoveFile={removeFile}
          disabled={busy && !micActive}
        />
      </div>

      <SettingsModal
        visible={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        initialPrompt={profile?.custom_system_prompt || ''}
        onSave={saveSystemPrompt}
      />
    </div>
  );
}
