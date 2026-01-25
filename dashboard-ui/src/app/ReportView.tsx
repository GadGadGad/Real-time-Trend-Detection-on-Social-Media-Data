"use client";

import { useState } from 'react';
import axios from 'axios';
import { FileText, Download, Play, CheckCircle } from 'lucide-react';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';

const API_BASE = "http://localhost:8000";

export default function ReportView() {
    const [report, setReport] = useState('');
    const [loading, setLoading] = useState(false);

    const generateReport = async () => {
        setLoading(true);
        try {
            const res = await axios.get(`${API_BASE}/report/generate`);
            setReport(res.data.report);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const handleDownload = () => {
        const element = document.createElement("a");
        const file = new Blob([report?.content || ""], { type: 'text/markdown' });
        element.href = URL.createObjectURL(file);
        element.download = `Report_22521060_${new Date().getTime()}.md`; // MSSV của Lê Minh Nhựt
        document.body.appendChild(element);
        element.click();
        document.body.removeChild(element);
    };

    return (
        <div className="h-full space-y-6 max-w-5xl mx-auto">
            <div className="flex justify-between items-center bg-slate-900/50 p-6 rounded-3xl border border-slate-800 backdrop-blur-sm">
                <div className="flex items-center gap-4">
                    <div className="p-3 bg-violet-600 rounded-2xl shadow-lg">
                        <FileText className="text-white" />
                    </div>
                    <div>
                        <h3 className="text-xl font-bold">Automated Strategy Synthesis</h3>
                        <p className="text-sm text-slate-500">AI-driven summary of top 15 trends and strategic recommendations.</p>
                    </div>
                </div>
                <button
                    onClick={generateReport}
                    disabled={loading}
                    className="flex items-center gap-2 px-6 py-3 bg-cyan-500 hover:bg-cyan-400 disabled:opacity-50 text-white rounded-xl font-bold transition-all shadow-lg"
                >
                    {loading ? (
                        <>
                            <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                            Analyzing...
                        </>
                    ) : (
                        <>
                            <Play size={18} fill="currentColor" />
                            Generate Report
                        </>
                    )}
                </button>
            </div>

            {report ? (
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="glass-panel rounded-3xl p-10 overflow-hidden relative"
                >
                    <div className="absolute top-0 right-0 p-6">
                    <button 
                        onClick={handleDownload}
                        className="flex items-center gap-2 px-6 py-3 bg-cyan-600 hover:bg-cyan-500 text-white rounded-xl font-bold transition-all shadow-lg shadow-cyan-500/20"
                    >
                        <Download size={18} />
                        Download .MD
                    </button>
                    </div>

                    <div className="prose prose-invert max-w-none bg-slate-900/40 p-8 rounded-3xl border border-slate-800">
                        <ReactMarkdown
                            components={{
                                h2: ({node, ...props}) => <h2 className="text-2xl font-bold text-cyan-400 mt-6 mb-4" {...props} />,
                                strong: ({node, ...props}) => <span className="font-bold text-cyan-300" {...props} />,
                                p: ({node, ...props}) => <p className="text-slate-300 leading-relaxed mb-4" {...props} />,
                                ul: ({node, ...props}) => <ul className="list-disc ml-6 space-y-2 mb-4 text-slate-300" {...props} />,
                                li: ({node, ...props}) => <li className="pl-2" {...props} />,
                            }}
                        >
                            {report || "Dữ liệu đang được phân tích..."}
                        </ReactMarkdown>
                    </div>
                </motion.div>
            ) : (
                <div className="flex flex-col items-center justify-center p-20 border-2 border-dashed border-slate-800 rounded-3xl text-slate-600">
                    <FileText size={48} className="mb-4 opacity-20" />
                    <p className="text-sm font-medium">Click generate to start the AI analysis pipeline.</p>
                </div>
            )}
        </div>
    );
}
