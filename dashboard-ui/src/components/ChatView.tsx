"use client";

import { useState } from 'react';
import axios from 'axios';
import { Send, Bot, User, Sparkles } from 'lucide-react';
import { motion } from 'framer-motion';

const API_BASE = "http://localhost:8000";

export default function ChatView() {
    const [messages, setMessages] = useState([
        { role: 'assistant', content: 'Chào Analyst! Tôi là Cyber Intelligence Agent. Bạn muốn tìm hiểu gì về luồng tin tức hôm nay?' }
    ]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);

    const sendMessage = async () => {
        if (!input.trim() || loading) return;

        const userMsg = { role: 'user', content: input };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setLoading(true);

        try {
            const res = await axios.post(`${API_BASE}/chat?query=${encodeURIComponent(input)}`);
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: res.data.answer,
                sources: res.data.sources
            }]);
        } catch (err) {
            setMessages(prev => [...prev, { role: 'assistant', content: '⚠️ Lỗi kết nối AI: ' + err.message }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-full max-w-4xl mx-auto space-y-4">
            <div className="flex-1 overflow-y-auto space-y-6 pr-4 scrollbar-hide">
                {messages.map((msg, i) => (
                    <motion.div
                        initial={{ opacity: 0, x: msg.role === 'user' ? 20 : -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        key={i}
                        className={`flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
                    >
                        <div className={`p-2 rounded-lg h-10 w-10 flex items-center justify-center shrink-0 shadow-lg ${msg.role === 'user' ? 'bg-violet-600' : 'bg-cyan-600'
                            }`}>
                            {msg.role === 'user' ? <User size={20} /> : <Bot size={20} />}
                        </div>

                        <div className={`max-w-[80%] space-y-2`}>
                            <div className={`p-4 rounded-2xl text-sm leading-relaxed shadow-xl ${msg.role === 'user'
                                    ? 'bg-slate-800 text-white rounded-tr-none border border-slate-700'
                                    : 'bg-slate-900 text-slate-200 rounded-tl-none border border-slate-800'
                                }`}>
                                {msg.content}
                            </div>

                            {msg.sources && msg.sources.length > 0 && (
                                <div className="flex flex-wrap gap-2 pt-1">
                                    {msg.sources.map((s: string, idx: number) => (
                                        <span key={idx} className="text-[10px] bg-slate-800/50 text-cyan-400 px-2 py-0.5 rounded border border-cyan-500/20">
                                            🔗 {s}
                                        </span>
                                    ))}
                                </div>
                            )}
                        </div>
                    </motion.div>
                ))}
                {loading && (
                    <div className="flex gap-4">
                        <div className="p-2 rounded-lg h-10 w-10 bg-cyan-600 flex items-center justify-center animate-pulse">
                            <Bot size={20} />
                        </div>
                        <div className="bg-slate-900 border border-slate-800 p-4 rounded-2xl rounded-tl-none animate-pulse">
                            <div className="flex gap-1">
                                <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-bounce"></div>
                                <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                                <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            <div className="relative group p-1 bg-gradient-to-r from-cyan-500 to-violet-500 rounded-2xl shadow-2xl">
                <div className="flex items-center bg-[#0f172a] rounded-[14px] p-2">
                    <input
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
                        placeholder="Search Intelligence Database..."
                        className="flex-1 bg-transparent border-none outline-none px-4 py-3 text-sm"
                    />
                    <button
                        onClick={sendMessage}
                        className="p-3 bg-cyan-500 hover:bg-cyan-400 text-white rounded-xl transition-all shadow-[0_0_15px_rgba(6,182,212,0.4)]"
                    >
                        <Send size={18} />
                    </button>
                </div>
            </div>
        </div>
    );
}
