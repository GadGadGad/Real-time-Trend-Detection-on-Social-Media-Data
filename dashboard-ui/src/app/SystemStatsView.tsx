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



                // Compute Taxonomy Distribution
                const taxonomyMap: Record<string, number> = {};
                trends.forEach((t: any) => {
                    const cat = t.category || 'Unclassified';
                    taxonomyMap[cat] = (taxonomyMap[cat] || 0) + 1;
                });
                const taxonomyData = Object.entries(taxonomyMap)
                    .map(([name, value]) => ({ name, value }))
                    .sort((a, b) => b.value - a.value);

                // Compute Source Breakdown
                const sourceMap: Record<string, number> = {
                    'GOOGLE TRENDS': 0,
                    'NEWS': 0,
                    'FACEBOOK': 0
                };
                trends.forEach((t: any) => {
                    if (t.representative_posts) {
                        t.representative_posts.forEach((p: any) => {
                            let src = p.source ? p.source.toUpperCase() : 'UNKNOWN';
                            if (src.includes('GOOGLE')) src = 'GOOGLE TRENDS';
                            else if (src.includes('FACEBOOK') || src === 'FB' || src === 'SOCIAL' || src.startsWith('FACE:') || src.startsWith('FACE ')) src = 'FACEBOOK';
                            else if (src.includes('NEWS') || src === 'TV' || src === 'WEB') src = 'NEWS';

                            if (sourceMap[src] !== undefined) {
                                sourceMap[src]++;
                            }
                        });
                    }
                });
                const sourceData = Object.entries(sourceMap)
                    .map(([name, value]) => ({ name, value }));

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
                    ],
                    taxonomyData,
                    sourceData
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
                        <h3 className="text-2xl font-bold font-mono">{stats?.totalPosts.toLocaleString()}</h3>
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
                        <h3 className="text-2xl font-bold font-mono">{stats?.totalTrends}</h3>
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

                {/* Taxonomy Distribution */}
                <div className="glass-panel p-6 rounded-3xl flex flex-col">
                    <h3 className="text-lg font-bold mb-6 flex items-center gap-2">
                        <PieChart size={18} className="text-violet-500" />
                        Taxonomy Distribution
                    </h3>
                    <div className="flex h-full">
                        <ResponsiveContainer width="60%" height="100%">
                            <PieChart>
                                <Pie
                                    data={stats?.taxonomyData}
                                    innerRadius={60}
                                    outerRadius={80}
                                    paddingAngle={5}
                                    dataKey="value"
                                >
                                    {stats?.taxonomyData.map((entry: any, index: number) => (
                                        <Cell key={`cell-${index}`} fill={['#ef4444', '#3b82f6', '#f59e0b', '#10b981', '#ec4899', '#8b5cf6', '#64748b'][index % 7]} />
                                    ))}
                                </Pie>
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '12px' }}
                                    itemStyle={{ color: '#fff' }}
                                    formatter={(value: number) => [`${value} trends`, 'Count']}
                                />
                            </PieChart>
                        </ResponsiveContainer>
                        <div className="flex flex-col justify-center gap-2 overflow-y-auto pr-2 custom-scrollbar max-h-[300px] w-[40%]">
                            {stats?.taxonomyData.map((entry: any, index: number) => (
                                <div key={index} className="flex items-center gap-2">
                                    <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: ['#ef4444', '#3b82f6', '#f59e0b', '#10b981', '#ec4899', '#8b5cf6', '#64748b'][index % 7] }}></div>
                                    <div className="min-w-0">
                                        <p className="text-xs font-bold truncate">{entry.name}</p>
                                        <p className="text-[10px] text-slate-500">{entry.value} items</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Source Breakdown (New) */}
                <div className="glass-panel p-6 rounded-3xl flex flex-col">
                    <h3 className="text-lg font-bold mb-6 flex items-center gap-2">
                        <Activity size={18} className="text-cyan-500" />
                        Intel Source Breakdown
                    </h3>
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={stats?.sourceData} layout="vertical" margin={{ left: 10 }}>
                            <XAxis type="number" hide />
                            <YAxis
                                dataKey="name"
                                type="category"
                                width={100}
                                tick={{ fill: '#94a3b8', fontSize: 10, fontWeight: 'bold' }}
                                axisLine={false}
                                tickLine={false}
                            />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '12px' }}
                                itemStyle={{ color: '#fff' }}
                                cursor={{ fill: '#334155', opacity: 0.2 }}
                            />
                            <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={24}>
                                {(stats?.sourceData || []).map((entry: any, index: number) => (
                                    <Cell key={`cell-${index}`} fill={
                                        entry.name === 'GOOGLE TRENDS' ? '#4285F4' :
                                            entry.name === 'FACEBOOK' ? '#1877F2' :
                                                entry.name === 'NEWS' ? '#F59E0B' :
                                                    '#06b6d4'
                                    } />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
}
