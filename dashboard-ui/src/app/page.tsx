"use client";

import { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Activity,
  LayoutDashboard,
  MessageSquare,
  Map as MapIcon,
  FileText,
  Settings,
  Bell,
  Search,
  Zap
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// Specialized Tabs
import ChatView from '@/components/ChatView';
import SemanticMap from '@/components/SemanticMap';
import ReportView from './ReportView';
import IntelView from './IntelView';
import SystemStatsView from './SystemStatsView';
import SettingsView from './SettingsView';

// API Configuration
const API_BASE = "http://localhost:8000";

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('live');
  const [selectedTrendId, setSelectedTrendId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [trends, setTrends] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await axios.get(`${API_BASE}/trends`);
        setTrends(res.data);
        setLoading(false);
      } catch (err) {
        console.error("Fetch error:", err);
        setLoading(false);
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex h-screen bg-[#020617] text-slate-200 overflow-hidden font-sans">
      {/* Sidebar */}
      <aside className="w-64 border-r border-slate-800 bg-[#020617] flex flex-col">
        <div className="p-6 flex items-center gap-3">
          <div className="p-2 bg-cyan-500 rounded-lg shadow-[0_0_15px_rgba(6,182,212,0.5)]">
            <Zap className="text-white fill-white" size={20} />
          </div>
          <h1 className="text-xl font-bold gradient-text tracking-tighter">CYBER INTEL</h1>
        </div>

        <nav className="flex-1 px-4 space-y-2 py-4">
          <NavItem
            icon={<LayoutDashboard size={20} />}
            label="Live Feed"
            active={activeTab === 'live'}
            onClick={() => setActiveTab('live')}
          />
          <NavItem
            icon={<Activity size={20} />}
            label="Intelligence"
            active={activeTab === 'intel'}
            onClick={() => setActiveTab('intel')}
          />
          <NavItem
            icon={<MapIcon size={20} />}
            label="Semantic Map"
            active={activeTab === 'map'}
            onClick={() => setActiveTab('map')}
          />
          <NavItem
            icon={<MessageSquare size={20} />}
            label="AI Agent Chat"
            active={activeTab === 'chat'}
            onClick={() => setActiveTab('chat')}
          />
          <NavItem
            icon={<FileText size={20} />}
            label="Daily Reports"
            active={activeTab === 'reports'}
            onClick={() => setActiveTab('reports')}
          />
          <NavItem
            icon={<Activity size={20} />} // Reusing Activity icon or finding another
            label="System Status"
            active={activeTab === 'stats'}
            onClick={() => setActiveTab('stats')}
          />
        </nav>

        <div className="p-4 border-t border-slate-800">
          <NavItem
            icon={<Settings size={20} />}
            label="System Settings"
            active={activeTab === 'settings'}
            onClick={() => setActiveTab('settings')}
          />
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col relative overflow-hidden">
        {/* Header */}
        <header className="h-16 border-b border-slate-800 bg-[#020617]/50 backdrop-blur-md flex items-center justify-between px-8 z-10">
          <div className="flex items-center gap-4 bg-slate-900/50 px-4 py-2 rounded-full border border-slate-800 w-96">
            <Search size={18} className="text-slate-500" />
            <input
              placeholder="Search trends, entities, or news..."
              className="bg-transparent border-none outline-none text-sm w-full"
            />
          </div>

          <div className="flex items-center gap-4">
            <div className="relative p-2 text-slate-400 hover:text-white cursor-pointer transition-colors">
              <Bell size={20} />
              <div className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full border-2 border-[#020617]"></div>
            </div>
            <div className="flex items-center gap-2 pl-4 border-l border-slate-800">
              <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-cyan-500 to-violet-500 shadow-md"></div>
              <span className="text-sm font-medium">Analyst Unit 01</span>
            </div>
          </div>
        </header>

        {/* Dynamic Canvas */}
        <div className="flex-1 overflow-y-auto p-8 relative">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
              className="h-full"
            >
              {activeTab === 'live' && (
                <LiveView
                  trends={trends}
                  loading={loading}
                  onSelectTrend={(id) => {
                    setSelectedTrendId(id);
                    setActiveTab('intel');
                  }}
                />
              )}
              {activeTab === 'map' && <SemanticMap />}
              {activeTab === 'chat' && <ChatView />}
              {activeTab === 'reports' && <ReportView />}
              {activeTab === 'intel' && <IntelView initialTrendId={selectedTrendId || undefined} />}
              {activeTab === 'stats' && <SystemStatsView />}
              {activeTab === 'settings' && <SettingsView />}
            </motion.div>
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}

function NavItem({ icon, label, active, onClick }: { icon: any, label: string, active?: boolean, onClick?: () => void }) {
  return (
    <div
      onClick={onClick}
      className={`
        flex items-center gap-4 px-4 py-3 rounded-xl cursor-pointer transition-all duration-200
        ${active
          ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20'
          : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'}
      `}
    >
      {icon}
      <span className="text-sm font-semibold">{label}</span>
      {active && (
        <motion.div
          layoutId="active-pill"
          className="ml-auto w-1.5 h-1.5 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.8)]"
        />
      )}
    </div>
  );
}

function LiveView({ trends, loading, onSelectTrend }: { trends: any[], loading: boolean, onSelectTrend: (id: number) => void }) {
  if (loading && trends.length === 0) return <div>Loading Intelligence Stream...</div>;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold tracking-tight">Real-time Stream</h2>
        <div className="flex items-center gap-2 text-xs font-bold text-cyan-500 px-3 py-1 bg-cyan-500/10 rounded-full border border-cyan-500/20 uppercase">
          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-ping"></div>
          Monitoring Live
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {trends.map((trend) => (
          <TrendCard key={trend.id} trend={trend} onClick={() => onSelectTrend(trend.id)} />
        ))}
      </div>
    </div>
  );
}

function TrendCard({ trend, onClick }: { trend: any, onClick: () => void }) {
  return (
    <motion.div
      onClick={onClick}
      whileHover={{ scale: 1.02 }}
      className="glass-panel p-5 rounded-2xl group cursor-pointer hover:border-cyan-500/30 transition-all glow-card"
    >
      <div className="flex justify-between items-start mb-4">
        <span className="text-xs font-bold bg-slate-800 text-slate-400 px-2 py-1 rounded-md uppercase tracking-wider">
          {trend.category || 'Discovery'}
        </span>
        <div className="flex flex-col items-end">
          <span className="text-cyan-400 font-mono text-sm font-bold">{trend.trend_score.toFixed(1)}</span>
          <span className="text-[10px] text-slate-500 uppercase font-black tracking-widest">Score</span>
        </div>
      </div>

      <h3 className="text-lg font-bold text-slate-100 mb-2 leading-tight group-hover:text-cyan-400 transition-colors">
        {trend.trend_name}
      </h3>

      <p className="text-sm text-slate-400 line-clamp-2 mb-4">
        {trend.summary || "Pending strategic analysis..."}
      </p>

      <div className="flex items-center justify-between pt-4 border-t border-slate-800">
        <div className="flex -space-x-2">
          {[1, 2, 3].map(i => (
            <div key={i} className="w-6 h-6 rounded-full bg-slate-800 border-2 border-[#020617] flex items-center justify-center text-[10px]">
              👤
            </div>
          ))}
          <span className="pl-4 text-[10px] text-slate-500 font-bold">+{trend.post_count} posts</span>
        </div>
        <span className="text-[10px] text-slate-600 font-mono italic">
          {new Date(trend.last_updated).toLocaleTimeString()}
        </span>
      </div>
    </motion.div>
  );
}
