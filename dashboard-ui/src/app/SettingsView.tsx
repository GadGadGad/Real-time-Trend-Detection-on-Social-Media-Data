"use client";

import { useState } from 'react';
import { Sliders, Save, RefreshCw } from 'lucide-react';

export default function SettingsView() {
    const [threshold, setThreshold] = useState(30);
    const [sensitivity, setSensitivity] = useState(0.4);

    return (
        <div className="h-full flex items-center justify-center">
            <div className="glass-panel p-10 rounded-3xl w-full max-w-2xl bg-gradient-to-br from-slate-900 via-slate-900 to-[#120a2e]">
                <div className="text-center mb-10">
                    <div className="w-20 h-20 bg-slate-800 rounded-3xl mx-auto flex items-center justify-center mb-4 shadow-xl border border-slate-700">
                        <Sliders size={40} className="text-cyan-400" />
                    </div>
                    <h2 className="text-2xl font-bold">System Configuration</h2>
                    <p className="text-slate-500">Tune the sensitivity of the AI injection pipeline.</p>
                </div>

                <div className="space-y-8">
                    <div>
                        <div className="flex justify-between mb-2">
                            <label className="font-bold text-sm text-slate-300">Trend Score Threshold</label>
                            <span className="font-mono text-cyan-400 font-bold">{threshold.toFixed(1)}</span>
                        </div>
                        <input
                            type="range"
                            min="0" max="100"
                            value={threshold}
                            onChange={(e) => setThreshold(Number(e.target.value))}
                            className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                        />
                        <p className="text-xs text-slate-500 mt-2">Events below this score will be classified as 'Monitoring Only'.</p>
                    </div>

                    <div>
                        <div className="flex justify-between mb-2">
                            <label className="font-bold text-sm text-slate-300">Semantic Sensitivity</label>
                            <span className="font-mono text-violet-400 font-bold">{sensitivity.toFixed(2)}</span>
                        </div>
                        <input
                            type="range"
                            min="0" max="1" step="0.05"
                            value={sensitivity}
                            onChange={(e) => setSensitivity(Number(e.target.value))}
                            className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-violet-500"
                        />
                        <p className="text-xs text-slate-500 mt-2">Minimum cosine similarity required for RAG context retrieval.</p>
                    </div>

                    <div className="flex gap-4 pt-4">
                        <button className="flex-1 py-4 bg-cyan-600 hover:bg-cyan-500 rounded-xl font-bold transition-all flex items-center justify-center gap-2 shadow-lg shadow-cyan-900/20">
                            <Save size={18} /> Save Configuration
                        </button>
                        <button className="px-6 py-4 bg-slate-800 hover:bg-slate-700 rounded-xl font-bold transition-all flex items-center justify-center gap-2">
                            <RefreshCw size={18} /> Reset
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
