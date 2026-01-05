"use client";

import { useEffect, useState } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { Server, Database, Activity, CheckCircle, Clock } from 'lucide-react';

const API_BASE = "http://localhost:8000";

export default function SystemStatsView() {
    const [stats, setStats] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchStats = async () => {
            try {
                // We'll calculate stats from the trends endpoint for now since there's no dedicated stats endpoint yet
                const res = await axios.get(`${API_BASE}/trends?limit=1000`);
                const trends = res.data;

                const totalPosts = trends.reduce((acc: number, t: any) => acc + t.post_count, 0);
                const activetrends = trends.filter((t: any) => t.trend_score > 50).length;

                // Top topics for bar chart
                const topTopics = [...trends]
                    .sort((a: any, b: any) => b.post_count - a.post_count)
                    .slice(0, 8)
                    .map((t: any) => ({ name: t.trend_name, value: t.post_count }));

                // Status distribution
                const analyzed = trends.filter((t: any) => t.summary && t.summary.length > 20).length;
                const pending = trends.length - analyzed;

                setStats({
                    totalTrends: trends.length,
                    totalPosts,
                    activetrends,
                    topTopics,
                    statusData: [
                        { name: 'Deep Analysis', value: analyzed },
                        { name: 'Fast Path', value: pending }
                    ]
                });
                setLoading(false);
            } catch (err) {
                console.error(err);
                setLoading(false);
            }
        };
        fetchStats();
    }, []);

    if (loading) return <div className="text-center p-10 text-cyan-500 animate-pulse">Scanning System Metrics...</div>;

    return (
        <div className="h-full space-y-6 overflow-y-auto pr-2 scrollbar-hide">
            {/* KPI Cards */}
            <div className="grid grid-cols-3 gap-6">
                <div className="glass-panel p-6 rounded-3xl flex items-center gap-4 relative overflow-hidden group hover:border-cyan-500/50 transition-all">
                    <div className="p-4 bg-cyan-500/10 rounded-2xl text-cyan-400 group-hover:scale-110 transition-transform">
                        <Database size={24} />
                    </div>
                    <div>
                        <p className="text-sm text-slate-500 font-bold uppercase tracking-wider">Total Ingested</p>
                        <h3 className="text-2xl font-bold font-mono">{stats.totalPosts.toLocaleString()}</h3>
                    </div>
                    <div className="absolute -right-6 -bottom-6 opacity-5 rotate-12">
                        <Database size={100} />
                    </div>
                </div>

                <div className="glass-panel p-6 rounded-3xl flex items-center gap-4 relative overflow-hidden group hover:border-violet-500/50 transition-all">
                    <div className="p-4 bg-violet-500/10 rounded-2xl text-violet-400 group-hover:scale-110 transition-transform">
                        <Activity size={24} />
                    </div>
                    <div>
                        <p className="text-sm text-slate-500 font-bold uppercase tracking-wider">Active Signals</p>
                        <h3 className="text-2xl font-bold font-mono">{stats.totalTrends}</h3>
                    </div>
                    <div className="absolute -right-6 -bottom-6 opacity-5 rotate-12">
                        <Activity size={100} />
                    </div>
                </div>

                <div className="glass-panel p-6 rounded-3xl flex items-center gap-4 relative overflow-hidden group hover:border-emerald-500/50 transition-all">
                    <div className="p-4 bg-emerald-500/10 rounded-2xl text-emerald-400 group-hover:scale-110 transition-transform">
                        <Server size={24} />
                    </div>
                    <div>
                        <p className="text-sm text-slate-500 font-bold uppercase tracking-wider">System Status</p>
                        <h3 className="text-2xl font-bold font-mono text-emerald-400">ONLINE</h3>
                    </div>
                    <div className="absolute -right-6 -bottom-6 opacity-5 rotate-12">
                        <Server size={100} />
                    </div>
                </div>
            </div>

            {/* Charts Row */}
            <div className="grid grid-cols-2 gap-6 h-[400px]">
                <div className="glass-panel p-6 rounded-3xl flex flex-col">
                    <h3 className="text-lg font-bold mb-6">Pipeline Load Distribution</h3>
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={stats.topTopics} layout="vertical" margin={{ left: 20 }}>
                            <XAxis type="number" hide />
                            <YAxis dataKey="name" type="category" width={150} tick={{ fill: '#94a3b8', fontSize: 10 }} />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '12px' }}
                                itemStyle={{ color: '#fff' }}
                            />
                            <Bar dataKey="value" fill="#3b82f6" radius={[0, 4, 4, 0]} barSize={20} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>

                <div className="glass-panel p-6 rounded-3xl flex flex-col">
                    <h3 className="text-lg font-bold mb-6">AI Processing Output</h3>
                    <div className="flex h-full">
                        <ResponsiveContainer width="50%" height="100%">
                            <PieChart>
                                <Pie
                                    data={stats.statusData}
                                    innerRadius={60}
                                    outerRadius={80}
                                    paddingAngle={5}
                                    dataKey="value"
                                >
                                    <Cell fill="#7c3aed" />
                                    <Cell fill="#334155" />
                                </Pie>
                                <Tooltip />
                            </PieChart>
                        </ResponsiveContainer>
                        <div className="flex flex-col justify-center gap-4">
                            <div className="flex items-center gap-3">
                                <div className="w-3 h-3 rounded-full bg-violet-600"></div>
                                <div>
                                    <p className="text-sm font-bold">Deep Analysis</p>
                                    <p className="text-xs text-slate-500">{stats.statusData[0].value} items</p>
                                </div>
                            </div>
                            <div className="flex items-center gap-3">
                                <div className="w-3 h-3 rounded-full bg-slate-700"></div>
                                <div>
                                    <p className="text-sm font-bold">Fast Path</p>
                                    <p className="text-xs text-slate-500">{stats.statusData[1].value} items</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
