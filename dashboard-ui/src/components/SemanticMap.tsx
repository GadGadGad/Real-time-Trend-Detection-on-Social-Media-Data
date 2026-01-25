"use client";

import { useEffect, useState } from 'react';
import axios from 'axios';
import {
    ScatterChart,
    Scatter,
    XAxis,
    YAxis,
    ZAxis,
    Tooltip,
    ResponsiveContainer,
    Cell,
    CartesianGrid
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
        <div className="h-full w-full glass-panel rounded-3xl p-6 flex flex-col relative overflow-hidden min-h-[750px]">
            
            {/* 1. KHUNG BO TRÒN: Đã được căn giữa tuyệt đối và mờ hơn để không rối mắt */}
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="w-[85%] h-[85%] rounded-full border border-slate-800/30 border-dashed"></div>
                <div className="absolute w-[55%] h-[55%] rounded-full border border-slate-800/15"></div>
            </div>
    
            <div className="relative z-10 flex flex-col h-full">
                {/* Header */}
                <div className="mb-4">
                    <h3 className="text-xl font-bold">2D Semantic Projection</h3>
                    <p className="text-xs text-slate-500 mt-1 uppercase tracking-widest font-black">Spatial Intelligence v1.0</p>
                </div>
    
                {/* 2. KHU VỰC BIỂU ĐỒ: Tăng margin bottom lên 60 để không bao giờ bị chạm vào Legend */}
                <div className="flex-1 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 40, right: 40, bottom: 60, left: 40 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} horizontal={false} />
                            
                            {/* 3. FIX TỌA ĐỘ: Dùng 'auto' để Recharts tự tính khoảng cách tối ưu, 
                                 nhưng thêm padding 20% để các điểm bung rộng ra */}
                            <XAxis 
                                type="number" 
                                dataKey="x" 
                                hide 
                                domain={['dataMin - 10', 'dataMax + 10']} 
                            />
                            <YAxis 
                                type="number" 
                                dataKey="y" 
                                hide 
                                domain={['dataMin - 10', 'dataMax + 10']} 
                            />
                            
                            <ZAxis type="number" dataKey="score" range={[100, 400]} />
                            
                            <Tooltip
                                cursor={{ strokeDasharray: '3 3' }}
                                content={({ active, payload }) => {
                                    if (active && payload && payload.length) {
                                        const data = payload[0].payload;
                                        return (
                                            <div className="bg-slate-900 border border-slate-700 p-3 rounded-xl shadow-2xl backdrop-blur-md">
                                                <p className="text-sm font-bold text-white mb-1">{data.name}</p>
                                                <div className="flex items-center gap-2">
                                                    <span className="text-[10px] uppercase font-black px-1.5 py-0.5 rounded" style={{ backgroundColor: (CATEGORY_COLORS[data.category] || '#64748b') + '44', color: CATEGORY_COLORS[data.category] }}>
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
    
                            {/* 4. ĐIỂM DỮ LIỆU: Thêm stroke để khi các điểm chồng lên nhau vẫn nhìn rõ */}
                            <Scatter name="Trends" data={data}>
                                {data.map((entry: any, index: number) => (
                                    <Cell 
                                        key={`cell-${index}`} 
                                        fill={CATEGORY_COLORS[entry.category] || '#64748b'} 
                                        stroke="#0f172a"
                                        strokeWidth={1}
                                    />
                                ))}
                            </Scatter>
                        </ScatterChart>
                    </ResponsiveContainer>
                </div>
    
                {/* 5. LEGEND: Đảm bảo có khoảng cách an toàn phía trên */}
                <div className="mt-auto pt-6 border-t border-slate-800 flex flex-wrap gap-4">
                    {Object.entries(CATEGORY_COLORS).slice(0, 8).map(([cat, color]: [string, any]) => (
                        <div key={cat} className="flex items-center gap-2">
                            <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }}></div>
                            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-tighter">{cat}</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
