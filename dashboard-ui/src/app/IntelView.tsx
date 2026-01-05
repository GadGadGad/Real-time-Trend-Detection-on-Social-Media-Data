"use client";

import { useState, useEffect } from 'react';
import axios from 'axios';
import {
    Activity,
    TrendingUp,
    Share2,
    MessageCircle,
    AlertTriangle,
    CheckCircle,
    BrainCircuit,
    ArrowRight
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RechartsTooltip } from 'recharts';

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

    return (
        <div className="flex h-full gap-6">
            {/* Left List */}
            <div className="w-1/3 flex flex-col gap-3 overflow-y-auto pr-2 scrollbar-hide">
                <h3 className="text-lg font-bold mb-2">Active Signals ({trends.length})</h3>
                {trends.map(trend => (
                    <div
                        key={trend.id}
                        onClick={() => setSelectedId(trend.id)}
                        className={`p-4 rounded-xl cursor-pointer border transition-all ${selectedId === trend.id
                                ? 'bg-cyan-500/10 border-cyan-500 shadow-[0_0_15px_rgba(6,182,212,0.2)]'
                                : 'bg-slate-900/50 border-slate-800 hover:border-slate-700'
                            }`}
                    >
                        <div className="flex justify-between items-start mb-2">
                            <span className="text-[10px] font-bold bg-slate-800 px-2 py-0.5 rounded text-slate-400 capitalize">
                                {trend.category || 'General'}
                            </span>
                            <span className="font-mono text-cyan-400 font-bold">{trend.trend_score.toFixed(1)}</span>
                        </div>
                        <h4 className={`font-bold text-sm line-clamp-2 ${selectedId === trend.id ? 'text-cyan-100' : 'text-slate-300'}`}>
                            {trend.trend_name}
                        </h4>
                    </div>
                ))}
            </div>

            {/* Right Detail Panel */}
            <AnimatePresence mode="wait">
                {selectedTrend && (
                    <motion.div
                        key={selectedTrend.id}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                        className="flex-1 overflow-y-auto pr-2 scrollbar-hide"
                    >
                        {/* Header */}
                        <div className="glass-panel p-6 rounded-3xl mb-6 relative overflow-hidden">
                            <div className="absolute top-0 right-0 p-4 opacity-10">
                                <Activity size={120} />
                            </div>
                            <div className="relative z-10">
                                <div className="flex items-center gap-3 mb-4">
                                    <span className="px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider"
                                        style={{ backgroundColor: CATEGORY_COLORS[selectedTrend.category] + '33', color: CATEGORY_COLORS[selectedTrend.category] }}>
                                        {selectedTrend.category || 'Unclassified'}
                                    </span>
                                    <span className="text-slate-500 text-xs font-mono">{new Date(selectedTrend.last_updated).toLocaleString()}</span>
                                </div>
                                <h1 className="text-3xl font-bold mb-4 leading-tight gradient-text">{selectedTrend.trend_name}</h1>

                                <div className="flex gap-8 border-t border-slate-800/50 pt-4">
                                    <div>
                                        <p className="text-xs text-slate-500 uppercase font-black">Impact Score</p>
                                        <p className="text-2xl font-mono font-bold text-cyan-400">{selectedTrend.trend_score.toFixed(1)}</p>
                                    </div>
                                    <div>
                                        <p className="text-xs text-slate-500 uppercase font-black">Volume</p>
                                        <p className="text-2xl font-mono font-bold text-violet-400">{selectedTrend.post_count}</p>
                                    </div>
                                    <div>
                                        <p className="text-xs text-slate-500 uppercase font-black">Sources</p>
                                        <p className="text-2xl font-mono font-bold text-emerald-400">Multi-channel</p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* AI Analysis */}
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                            <div className="glass-panel p-6 rounded-3xl flex flex-col">
                                <div className="flex items-center gap-2 mb-4 text-violet-400">
                                    <BrainCircuit size={20} />
                                    <h3 className="font-bold uppercase tracking-wider text-sm">AI Executive Summary</h3>
                                </div>
                                <p className="text-slate-300 leading-relaxed text-sm flex-1">
                                    {selectedTrend.summary || "Pending algorithmic generation..."}
                                </p>
                            </div>

                            <div className="glass-panel p-6 rounded-3xl">
                                <div className="flex items-center gap-2 mb-4 text-emerald-400">
                                    <CheckCircle size={20} />
                                    <h3 className="font-bold uppercase tracking-wider text-sm">Strategic Recommendations</h3>
                                </div>
                                <div className="space-y-4">
                                    <div className="bg-slate-900/50 p-4 rounded-xl border-l-2 border-emerald-500">
                                        <p className="text-xs font-bold text-emerald-500 mb-1">FOR GOVERNMENT</p>
                                        <p className="text-sm text-slate-300">Monitor public sentiment evolution; prepare official statement if semantic intensity crosses 75.0 threshold.</p>
                                    </div>
                                    <div className="bg-slate-900/50 p-4 rounded-xl border-l-2 border-blue-500">
                                        <p className="text-xs font-bold text-blue-500 mb-1">FOR BUSINESS</p>
                                        <p className="text-sm text-slate-300">Assess supply chain impact key entities mentioned in the cluster.</p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Related Trends */}
                        {related.length > 0 && (
                            <div className="mb-6">
                                <h3 className="font-bold text-slate-400 mb-3 flex items-center gap-2">
                                    <Share2 size={16} /> Related Signals (RAG)
                                </h3>
                                <div className="grid grid-cols-2 gap-4">
                                    {related.map((rt: any) => (
                                        <div key={rt.trend_name} className="bg-slate-900/50 p-3 rounded-xl border border-slate-800 flex items-center justify-between group cursor-pointer hover:border-cyan-500/30 transition-all">
                                            <span className="text-sm font-medium text-slate-300 group-hover:text-cyan-400 transition-colors line-clamp-1">{rt.trend_name || rt.name}</span>
                                            <span className="text-xs font-mono text-slate-500">{(rt.score * 100).toFixed(0)}% Match</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Representative Posts */}
                        <div>
                            <h3 className="font-bold text-slate-400 mb-3 flex items-center gap-2">
                                <MessageCircle size={16} /> Key Narratives
                            </h3>
                            <div className="space-y-3">
                                {(selectedTrend.representative_posts || []).slice(0, 3).map((post: any, i: number) => (
                                    <div key={i} className="bg-slate-900 p-4 rounded-xl border-l-4 border-l-cyan-500 shadow-md">
                                        <div className="flex justify-between items-center mb-2">
                                            <span className="text-xs font-bold text-cyan-500">{post.source}</span>
                                            <span className="text-[10px] text-slate-600 font-mono">{post.time}</span>
                                        </div>
                                        <p className="text-sm text-slate-300 leading-relaxed">{post.content}</p>
                                    </div>
                                ))}
                            </div>
                        </div>

                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
