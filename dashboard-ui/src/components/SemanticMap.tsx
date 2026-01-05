"use client";

import { useEffect, useState } from 'react';
import axios from 'axios';
import {
    ScatterChart,
    Scatter,
    XAxis,
    YAxis,
    ZAxis, f
    Tooltip,
    ResponsiveContainer,
    Cell
} from 'recharts';

const API_BASE = "http://localhost:8000";

const CATEGORY_COLORS: any = {
    'T1': '#ef4444', 'T2': '#3b82f6', 'T3': '#f59e0b',
    'T4': '#10b981', 'T5': '#ec4899', 'T6': '#8b5cf6', 'T7': '#64748b', 'Other': '#64748b'
};

export default function SemanticMap() {
    const [data, setData] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const res = await axios.get(`${API_BASE}/analytics/semantic-map`);
                setData(res.data.data);
            } catch (err) {
                console.error(err);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, []);

    if (loading) return <div className="flex items-center justify-center h-full text-cyan-500 animate-pulse">Calculating Field Coordinates...</div>;

    return (
        <div className="h-full w-full glass-panel rounded-3xl p-6 flex flex-col">
            <div className="mb-6">
                <h3 className="text-xl font-bold">2D Semantic Projection</h3>
                <p className="text-xs text-slate-500 mt-1 uppercase tracking-widest font-black">Spatial Intelligence v1.0</p>
            </div>

            <div className="flex-1 min-h-[500px]">
                <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <XAxis type="number" dataKey="x" hide />
                        <YAxis type="number" dataKey="y" hide />
                        <ZAxis type="number" dataKey="score" range={[50, 400]} />
                        <Tooltip
                            cursor={{ strokeDasharray: '3 3' }}
                            content={({ active, payload }) => {
                                if (active && payload && payload.length) {
                                    const data = payload[0].payload;
                                    return (
                                        <div className="bg-slate-900 border border-slate-700 p-3 rounded-xl shadow-2xl backdrop-blur-md">
                                            <p className="text-sm font-bold text-white mb-1">{data.name}</p>
                                            <div className="flex items-center gap-2">
                                                <span className="text-[10px] uppercase font-black px-1.5 py-0.5 rounded" style={{ backgroundColor: CATEGORY_COLORS[data.category] + '44', color: CATEGORY_COLORS[data.category] }}>
                                                    {data.category}
                                                </span>
                                                <span className="text-[10px] text-slate-400 font-mono">Intensity: {data.score.toFixed(1)}</span>
                                            </div>
                                        </div>
                                    );
                                }
                                return null;
                            }}
                        />
                        <Scatter name="Trends" data={data}>
                            {data.map((entry: any, index: number) => (
                                <Cell key={`cell-${index}`} fill={CATEGORY_COLORS[entry.category] || '#64748b'} strokeWidth={2} strokeOpacity={0.8} />
                            ))}
                        </Scatter>
                    </ScatterChart>
                </ResponsiveContainer>
            </div>

            <div className="mt-4 flex flex-wrap gap-4 pt-4 border-t border-slate-800">
                {Object.entries(CATEGORY_COLORS).slice(0, 7).map(([cat, color]: [string, any]) => (
                    <div key={cat} className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }}></div>
                        <span className="text-[10px] font-bold text-slate-500 uppercase">{cat}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}
