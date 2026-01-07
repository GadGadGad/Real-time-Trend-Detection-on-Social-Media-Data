"use client";

import { useState, useEffect } from 'react';
import axios from 'axios';
import {
    Activity,
    TrendingUp,
    Share2,
    MessageCircle,
    CheckCircle,
    BrainCircuit,
    Download
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE = "http://localhost:8000";

const CATEGORY_COLORS: any = {
    'T1': '#ef4444', 'T2': '#3b82f6', 'T3': '#f59e0b',
    'T4': '#10b981', 'T5': '#ec4899', 'T6': '#8b5cf6', 'T7': '#64748b'
};

export default function IntelView({ initialTrendId }: { initialTrendId?: number }) {
    const [trends, setTrends] = useState<any[]>([]);
    const [selectedId, setSelectedId] = useState<number | null>(initialTrendId || null);
    const [related, setRelated] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [sortBy, setSortBy] = useState<'score' | 'time'>('score');
    const [currentPage, setCurrentPage] = useState(1);
    const itemsPerPage = 10;

    useEffect(() => {
        const fetchTrends = async () => {
            try {
                const res = await axios.get(`${API_BASE}/trends?limit=50`);
                setTrends(res.data);
                if (!selectedId && res.data.length > 0) {
                    setSelectedId(res.data[0].id);
                }
                setLoading(false);
            } catch (err) {
                console.error(err);
                setLoading(false);
            }
        };
        fetchTrends();
    }, []);

    useEffect(() => {
        if (!selectedId) return;

        const fetchRelated = async () => {
            try {
                const res = await axios.get(`${API_BASE}/trends/${selectedId}/related`);
                setRelated(res.data);
            } catch (err) {
                console.error("Failed to fetch related trends", err);
                setRelated([]);
            }
        };
        fetchRelated();
    }, [selectedId]);

    const selectedTrend = trends.find(t => t.id === selectedId);

    if (loading) return <div className="text-center p-10 animate-pulse text-cyan-500">Loading Intelligence Data...</div>;

    // ... (fetch effects remain the same)

    // Derived state
    const sortedTrends = [...trends].sort((a, b) => {
        if (sortBy === 'score') return b.trend_score - a.trend_score;
        return new Date(b.last_updated).getTime() - new Date(a.last_updated).getTime();
    });

    const totalPages = Math.ceil(sortedTrends.length / itemsPerPage);
    const visibleTrends = sortedTrends.slice(
        (currentPage - 1) * itemsPerPage,
        currentPage * itemsPerPage
    );

    const nextPage = () => setCurrentPage(p => Math.min(p + 1, totalPages));
    const prevPage = () => setCurrentPage(p => Math.max(p - 1, 1));

    if (loading) return <div className="text-center p-10 animate-pulse text-cyan-500">Loading Intelligence Data...</div>;

    return (
        <div className="flex h-full gap-6 relative overflow-hidden">
            {/* Watermark */}
            <div className="absolute inset-0 pointer-events-none flex items-center justify-center opacity-[0.03] z-0">
                <h1 className="text-[150px] font-black -rotate-45 text-slate-900 dark:text-slate-100 whitespace-nowrap">
                    CONFIDENTIAL // EYES ONLY
                </h1>
            </div>

            {/* Left List */}
            <div className="w-1/3 flex flex-col gap-3 z-10 h-full">
                <div className="flex items-center justify-between mb-2 pr-2">
                    <h3 className="text-lg font-bold flex items-center gap-2">
                        Active Signals <span className="text-xs font-mono text-slate-500">({trends.length})</span>
                    </h3>
                    <div className="flex bg-slate-100 dark:bg-slate-800 rounded-lg p-1">
                        <button
                            onClick={() => setSortBy('score')}
                            className={`px-2 py-1 text-[10px] font-bold rounded-md transition-all ${sortBy === 'score' ? 'bg-white dark:bg-slate-700 shadow text-cyan-600' : 'text-slate-400 hover:text-slate-600'}`}
                        >
                            SCORE
                        </button>
                        <button
                            onClick={() => setSortBy('time')}
                            className={`px-2 py-1 text-[10px] font-bold rounded-md transition-all ${sortBy === 'time' ? 'bg-white dark:bg-slate-700 shadow text-cyan-600' : 'text-slate-400 hover:text-slate-600'}`}
                        >
                            TIME
                        </button>
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto pr-2 scrollbar-hide space-y-3">
                    {visibleTrends.map(trend => (
                        <div
                            key={trend.id}
                            onClick={() => setSelectedId(trend.id)}
                            title={trend.trend_name}
                            className={`p-4 rounded-xl cursor-pointer border transition-all ${selectedId === trend.id
                                ? 'bg-cyan-500/10 border-cyan-600 dark:border-cyan-500 shadow-[0_0_15px_rgba(6,182,212,0.2)]'
                                : 'bg-white dark:bg-slate-900/50 border-slate-200 dark:border-slate-800 hover:border-slate-400 dark:hover:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800/50'
                                }`}
                        >
                            <div className="flex justify-between items-start mb-2">
                                <span className="text-[10px] font-bold bg-slate-800 px-2 py-0.5 rounded text-slate-400 capitalize">
                                    {trend.category || 'General'}
                                </span>
                                <span className="font-mono text-cyan-400 font-bold">{trend.trend_score.toFixed(1)}</span>
                            </div>
                            <h4 className={`font-bold text-sm ${selectedId === trend.id ? 'text-cyan-100' : 'text-slate-300'}`}>
                                {trend.trend_name}
                            </h4>
                        </div>
                    ))}
                </div>

                {/* Pagination Controls */}
                <div className="flex items-center justify-between pt-2 border-t border-slate-200 dark:border-slate-800 pr-2">
                    <button
                        onClick={prevPage}
                        disabled={currentPage === 1}
                        className="px-3 py-1 text-xs font-bold rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-500 disabled:opacity-30 disabled:cursor-not-allowed hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
                    >
                        PREV
                    </button>
                    <span className="text-xs font-mono text-slate-400">
                        PAGE {currentPage} / {totalPages}
                    </span>
                    <button
                        onClick={nextPage}
                        disabled={currentPage === totalPages}
                        className="px-3 py-1 text-xs font-bold rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-500 disabled:opacity-30 disabled:cursor-not-allowed hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
                    >
                        NEXT
                    </button>
                </div>
            </div>

            {/* Right Detail Panel - The Dossier */}
            <AnimatePresence mode="wait">
                {selectedTrend && (
                    <motion.div
                        key={selectedTrend.id}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                        className="flex-1 overflow-y-auto pr-2 scrollbar-hide z-10"
                    >
                        {/* Dossier Header / HUD */}
                        <div className="glass-panel p-1 rounded-3xl mb-6 relative overflow-visible border border-slate-200 dark:border-slate-700/50 bg-white/50 dark:bg-slate-900/50 backdrop-blur-xl">
                            {/* Decorative Tech Lines */}
                            <div className="absolute -top-3 left-10 right-10 h-[1px] bg-gradient-to-r from-transparent via-cyan-500/50 to-transparent"></div>
                            <div className="absolute top-0 left-10 w-[1px] h-3 bg-cyan-500/50"></div>
                            <div className="absolute top-0 right-10 w-[1px] h-3 bg-cyan-500/50"></div>

                            <div className="p-6 md:p-8">
                                <div className="flex items-start justify-between mb-8">
                                    <div>
                                        <div className="flex items-center gap-3 mb-2">
                                            <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-widest border border-current"
                                                style={{ color: CATEGORY_COLORS[selectedTrend.category], borderColor: CATEGORY_COLORS[selectedTrend.category] }}>
                                                {selectedTrend.category || 'Unclassified'}
                                            </span>
                                            <span className="text-slate-400 text-[10px] font-mono tracking-widest">
                                                ID: {selectedTrend.id}-{new Date(selectedTrend.last_updated).getTime().toString().slice(-6)}
                                            </span>
                                        </div>
                                        <h1 className="text-3xl md:text-4xl font-black uppercase tracking-tight text-slate-900 dark:text-slate-100 mb-2">
                                            {selectedTrend.trend_name}
                                        </h1>
                                        <div className="flex items-center gap-2 text-cyan-600 dark:text-cyan-400 text-xs font-mono">
                                            <div className="w-2 h-2 rounded-full bg-cyan-500 animate-pulse"></div>
                                            LIVE MONITORING ACTIVE
                                        </div>
                                    </div>
                                    <button
                                        onClick={() => window.print()}
                                        className="p-3 bg-slate-100 dark:bg-slate-800 hover:bg-cyan-500 hover:text-white rounded-xl text-slate-500 transition-all no-print group"
                                        title="Export Dossier"
                                    >
                                        <Download size={20} className="group-hover:scale-110 transition-transform" />
                                    </button>
                                </div>

                                {/* HUD Grid */}
                                <div className="grid grid-cols-3 gap-6 pt-6 border-t border-slate-200 dark:border-slate-800">
                                    <div className="relative group">
                                        <div className="text-[10px] text-slate-400 font-bold uppercase tracking-widest mb-1">Impact Score</div>
                                        <div className="text-3xl font-mono font-bold text-slate-900 dark:text-slate-100 flex items-baseline gap-2">
                                            {selectedTrend.trend_score.toFixed(1)}
                                            <span className="text-xs text-emerald-500 font-sans font-bold flex items-center">
                                                <TrendingUp size={12} className="mr-0.5" /> +2.4%
                                            </span>
                                        </div>
                                        <div className="h-1 w-full bg-slate-200 dark:bg-slate-800 mt-2 rounded-full overflow-hidden">
                                            <div className="h-full bg-cyan-500" style={{ width: `${Math.min(selectedTrend.trend_score * 10, 100)}%` }}></div>
                                        </div>
                                    </div>

                                    <div className="relative">
                                        <div className="text-[10px] text-slate-400 font-bold uppercase tracking-widest mb-1">Signal Volume</div>
                                        <div className="text-3xl font-mono font-bold text-slate-900 dark:text-slate-100">
                                            {selectedTrend.post_count}
                                        </div>
                                        <div className="text-[10px] text-slate-500 font-mono mt-1">
                                            SOURCES: GOOGLE TRENDS, NEWS, FACEBOOK
                                        </div>
                                    </div>

                                    <div className="relative">
                                        <div className="text-[10px] text-slate-400 font-bold uppercase tracking-widest mb-1">Sentiment Spectrum</div>
                                        <div className="h-4 w-full bg-gradient-to-r from-red-500 via-slate-400 to-emerald-500 rounded-sm mt-2 relative">
                                            {/* Pointer */}
                                            <div className="absolute -top-1 bottom-[-4px] w-1 bg-white border border-slate-900 shadow-sm" style={{ left: '65%' }}></div>
                                        </div>
                                        <div className="flex justify-between text-[8px] text-slate-500 font-mono mt-1 uppercase">
                                            <span>Negative</span>
                                            <span className="text-emerald-500 font-bold">Positive Leaning</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* AI Briefing */}
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                            <div className="glass-panel p-1 rounded-3xl h-full flex flex-col bg-slate-950 border border-slate-800">
                                <div className="p-6 flex flex-col h-full">
                                    <div className="flex items-center gap-3 mb-4 text-violet-400 pb-4 border-b border-violet-500/20">
                                        <BrainCircuit size={20} />
                                        <h3 className="font-bold uppercase tracking-widest text-xs">AI Executive Summary</h3>
                                    </div>
                                    <div className="font-mono text-slate-300 text-sm leading-relaxed flex-1">
                                        <TypewriterText text={selectedTrend.summary || "Pending algorithmic generation..."} />
                                    </div>
                                </div>
                            </div>

                            <div className="glass-panel p-6 rounded-3xl border border-emerald-500/20 bg-emerald-950/10">
                                <div className="flex items-center gap-3 mb-4 text-emerald-400">
                                    <CheckCircle size={20} />
                                    <h3 className="font-bold uppercase tracking-widest text-xs">Strategic Action Plan</h3>
                                </div>
                                <div className="space-y-4">
                                    <div className="bg-slate-900/60 p-4 rounded-xl border-l-2 border-emerald-500">
                                        <p className="text-[10px] font-bold text-emerald-500 mb-1 uppercase tracking-wider">Public Sector</p>
                                        <p className="text-sm text-slate-300">{selectedTrend.advice_state || "Monitoring public sentiment evolution..."}</p>
                                    </div>
                                    <div className="bg-slate-900/60 p-4 rounded-xl border-l-2 border-blue-500">
                                        <p className="text-[10px] font-bold text-blue-500 mb-1 uppercase tracking-wider">Private Sector</p>
                                        <p className="text-sm text-slate-300">{selectedTrend.advice_business || "Assessing supply chain impact..."}</p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Related Trends (RAG) */}
                        {related.length > 0 && (
                            <div className="mb-6">
                                <h3 className="font-bold text-slate-400 mb-3 flex items-center gap-2 text-sm uppercase tracking-wider">
                                    <Share2 size={14} /> Correlated Signals
                                </h3>
                                <div className="grid grid-cols-2 gap-4">
                                    {related.map((rt: any, i: number) => (
                                        <div
                                            key={`${rt.trend_name}-${i}`}
                                            title={rt.trend_name || rt.name}
                                            className="bg-slate-900/50 p-3 rounded-lg border border-slate-800 flex items-center justify-between group cursor-pointer hover:border-cyan-500/30 transition-all"
                                        >
                                            <span className="text-sm font-medium text-slate-300 group-hover:text-cyan-400 transition-colors line-clamp-1">
                                                {rt.trend_name || rt.name}
                                            </span>
                                            <span className="text-xs font-mono text-cyan-500/70 ml-2">
                                                {(rt.score * 100).toFixed(0)}%
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Evidence Wrapper */}
                        <div>
                            <h3 className="font-bold text-slate-400 mb-3 flex items-center gap-2 text-sm uppercase tracking-wider">
                                <MessageCircle size={14} /> Raw Evidence (Intercepted)
                            </h3>
                            <div className="space-y-3">
                                {(selectedTrend.representative_posts || []).slice(0, 3).map((post: any, i: number) => (
                                    <div key={i} className="bg-slate-50 dark:bg-slate-900 p-4 rounded-none border-l-2 border-slate-300 dark:border-slate-700 font-mono text-xs">
                                        <div className="flex justify-between items-center mb-1 text-slate-400">
                                            <span>SOURCE: {post.source?.toUpperCase() || 'UNKNOWN'}</span>
                                            <span>{post.time}</span>
                                        </div>
                                        <p className="text-slate-700 dark:text-slate-300">{post.content}</p>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Print Styles */}
                        <style jsx global>{`
                            @media print {
                                aside, header, .no-print { display: none !important; }
                                main { overflow: visible !important; height: auto !important; }
                                .glass-panel { break-inside: avoid; border: 1px solid #000; box-shadow: none; background: white; color: black; }
                                body { background: white; color: black; }
                                .text-slate-300, .text-slate-400, .text-slate-500 { color: #333 !important; }
                                * { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
                            }
                        `}</style>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

function TypewriterText({ text }: { text: string }) {
    const [displayedText, setDisplayedText] = useState('');

    useEffect(() => {
        setDisplayedText(''); // Reset on text change
        let i = 0;
        const length = text.length;

        const timer = setInterval(() => {
            if (i < length) {
                setDisplayedText(text.slice(0, i + 1));
                i++;
            } else {
                clearInterval(timer);
            }
        }, 15);

        return () => clearInterval(timer);
    }, [text]);

    return (
        <span>
            {displayedText}
            <span className="animate-pulse inline-block w-1.5 h-4 bg-cyan-500 ml-1 align-middle"></span>
        </span>
    );
}
