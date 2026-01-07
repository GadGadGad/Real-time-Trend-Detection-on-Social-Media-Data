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
  Zap,
  Filter,
  ArrowUpDown,
  X
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// Specialized Tabs
import { LineChart, Line, ResponsiveContainer } from 'recharts';
import ChatView from '@/components/ChatView';
import SemanticMap from '@/components/SemanticMap';
import { ToastProvider, useToast } from '@/components/Toast';
import ReportView from './ReportView';
import IntelView from './IntelView';
import SystemStatsView from './SystemStatsView';
import SettingsView from './SettingsView';
import Skeleton from '@/components/Skeleton'; // Ready for next step

// API Configuration
const API_BASE = "http://localhost:8000";

export default function Dashboard() {
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const [activeTab, setActiveTab] = useState('live');
  const [selectedTrendId, setSelectedTrendId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [trends, setTrends] = useState<any[]>([]);
  const [filterCategory, setFilterCategory] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<'score' | 'time'>('score');
  const [isSidebarOpen, setIsSidebarOpen] = useState(false); // Mobile sidebar state

  // Initialize theme with safe storage access
  useEffect(() => {
    // ... theme logic ...
    if (typeof window !== 'undefined') {
      // ... existing theme logic ...
      try {
        const saved = localStorage.getItem('theme') as 'dark' | 'light';
        const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        let initialTheme: 'dark' | 'light' = 'dark';
        if (saved) initialTheme = saved;
        else if (systemDark) initialTheme = 'dark';
        setTheme(initialTheme);
        document.documentElement.classList.toggle('dark', initialTheme === 'dark');
      } catch (e) {
        setTheme('dark');
        document.documentElement.classList.add('dark');
      }
    }
  }, []);

  const toggleTheme = () => {
    // ... existing toggle logic ...
    const newTheme = theme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
    document.documentElement.classList.toggle('dark', newTheme === 'dark');
    try { localStorage.setItem('theme', newTheme); } catch (e) { }
  };

  // ... (derived state, effects remain same) ...
  const filteredTrends = trends
    .filter(t => !filterCategory || t.category === filterCategory)
    .sort((a, b) => {
      if (sortBy === 'score') return b.trend_score - a.trend_score;
      if (sortBy === 'time') return new Date(b.last_updated).getTime() - new Date(a.last_updated).getTime();
      return 0;
    });

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
    <ToastProvider>
      <div className="flex h-screen bg-slate-50 dark:bg-[#020617] text-slate-900 dark:text-slate-200 overflow-hidden font-sans transition-colors duration-300">

        {/* Mobile Sidebar Overlay */}
        <AnimatePresence>
          {isSidebarOpen && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsSidebarOpen(false)}
              className="fixed inset-0 bg-black/60 z-40 md:hidden backdrop-blur-sm"
            />
          )}
        </AnimatePresence>

        {/* Sidebar */}
        <aside className={`
          fixed md:static inset-y-0 left-0 z-50 w-64 border-r border-slate-200 dark:border-slate-800 bg-white dark:bg-[#020617] flex flex-col transition-transform duration-300 transform md:transform-none
          ${isSidebarOpen ? 'translate-x-0 shadow-2xl' : '-translate-x-full md:translate-x-0'}
        `}>
          <div className="p-6 flex items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-cyan-600 dark:bg-cyan-500 rounded-lg shadow-lg dark:shadow-[0_0_15px_rgba(6,182,212,0.5)]">
                <Zap className="text-white fill-white" size={20} />
              </div>
              <h1 className="text-xl font-bold gradient-text tracking-tighter">CYBER INTEL</h1>
            </div>
            {/* Mobile Close Button */}
            <button onClick={() => setIsSidebarOpen(false)} className="md:hidden text-slate-500 hover:text-slate-800 dark:hover:text-white">
              <X size={20} />
            </button>
          </div>

          <nav className="flex-1 px-4 space-y-2 py-4">
            {/* ... nav items ... */}
            <NavItem
              icon={<LayoutDashboard size={20} />}
              label="Live Feed"
              active={activeTab === 'live'}
              onClick={() => { setActiveTab('live'); setIsSidebarOpen(false); }}
            />
            <NavItem
              icon={<Activity size={20} />}
              label="Intelligence"
              active={activeTab === 'intel'}
              onClick={() => { setActiveTab('intel'); setIsSidebarOpen(false); }}
            />
            <NavItem
              icon={<MapIcon size={20} />}
              label="Semantic Map"
              active={activeTab === 'map'}
              onClick={() => { setActiveTab('map'); setIsSidebarOpen(false); }}
            />
            <NavItem
              icon={<MessageSquare size={20} />}
              label="AI Agent Chat"
              active={activeTab === 'chat'}
              onClick={() => { setActiveTab('chat'); setIsSidebarOpen(false); }}
            />
            <NavItem
              icon={<FileText size={20} />}
              label="Daily Reports"
              active={activeTab === 'reports'}
              onClick={() => { setActiveTab('reports'); setIsSidebarOpen(false); }}
            />
            <NavItem
              icon={<Activity size={20} />}
              label="System Status"
              active={activeTab === 'stats'}
              onClick={() => { setActiveTab('stats'); setIsSidebarOpen(false); }}
            />
          </nav>

          <div className="p-4 border-t border-slate-200 dark:border-slate-800">
            <button
              onClick={toggleTheme}
              className="w-full flex items-center gap-4 px-4 py-3 rounded-xl cursor-pointer text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800/50 hover:text-slate-900 dark:hover:text-slate-200 transition-all mb-2"
            >
              {theme === 'dark' ? <Settings size={20} /> : <Zap size={20} />}
              <span className="text-sm font-semibold">{theme === 'dark' ? 'Light Mode' : 'Dark Mode'}</span>
            </button>
            <NavItem
              icon={<Settings size={20} />}
              label="System Settings"
              active={activeTab === 'settings'}
              onClick={() => { setActiveTab('settings'); setIsSidebarOpen(false); }}
            />
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 flex flex-col relative overflow-hidden transition-colors duration-300">
          {/* Header */}
          <header className="h-16 border-b border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-[#020617]/50 backdrop-blur-md flex items-center justify-between px-4 md:px-8 z-10 transition-colors duration-300">
            <div className="flex items-center gap-4">
              {/* Mobile Menu Button */}
              <button
                onClick={() => setIsSidebarOpen(true)}
                className="md:hidden p-2 -ml-2 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg"
                aria-label="Open Sidebar"
              >
                <div className="space-y-1.5">
                  <span className="block w-6 h-0.5 bg-current"></span>
                  <span className="block w-6 h-0.5 bg-current"></span>
                  <span className="block w-6 h-0.5 bg-current"></span>
                </div>
              </button>

              <div className="flex items-center gap-4 bg-slate-100 dark:bg-slate-900/50 px-4 py-2 rounded-full border border-slate-200 dark:border-slate-800 w-full max-w-xs md:max-w-md hidden md:flex">
                <Search size={18} className="text-slate-400 dark:text-slate-500" />
                <input
                  placeholder="Search trends..."
                  className="bg-transparent border-none outline-none text-sm w-full dark:text-slate-200 text-slate-800 placeholder:text-slate-400 dark:placeholder:text-slate-500"
                />
              </div>
            </div>

            <div className="flex items-center gap-4">
              <div className="relative p-2 text-slate-400 hover:text-cyan-600 dark:hover:text-white cursor-pointer transition-colors">
                <Bell size={20} />
                <div className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full border-2 border-white dark:border-[#020617]"></div>
              </div>
              <div className="flex items-center gap-2 pl-4 border-l border-slate-200 dark:border-slate-800">
                <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-cyan-500 to-violet-500 shadow-md"></div>
                <span className="text-sm font-medium text-slate-700 dark:text-slate-200">Analyst Unit 01</span>
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
                    trends={filteredTrends}
                    loading={loading}
                    onSelectTrend={(id) => {
                      setSelectedTrendId(id);
                      setActiveTab('intel');
                    }}
                    activeCategory={filterCategory}
                    onFilterCategory={setFilterCategory}
                    sortBy={sortBy}
                    onSortBy={setSortBy}
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
    </ToastProvider>
  );
}

function NavItem({ icon, label, active, onClick }: { icon: any, label: string, active?: boolean, onClick?: () => void }) {
  return (
    <div
      onClick={onClick}
      className={`
        flex items-center gap-4 px-4 py-3 rounded-xl cursor-pointer transition-all duration-200
        ${active
          ? 'bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 border border-cyan-500/20'
          : 'text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800/50 hover:text-slate-900 dark:hover:text-slate-200'}
      `}
    >
      {icon}
      <span className="text-sm font-semibold">{label}</span>
      {active && (
        <motion.div
          layoutId="active-pill"
          className="ml-auto w-1.5 h-1.5 rounded-full bg-cyan-500 dark:bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.8)]"
        />
      )}
    </div>
  );
}

function LiveView({
  trends,
  loading,
  onSelectTrend,
  activeCategory,
  onFilterCategory,
  sortBy,
  onSortBy
}: {
  trends: any[],
  loading: boolean,
  onSelectTrend: (id: number) => void,
  activeCategory: string | null,
  onFilterCategory: (c: string | null) => void,
  sortBy: 'score' | 'time',
  onSortBy: (s: 'score' | 'time') => void
}) {
  // Derive unique categories from actual data
  const categories = Array.from(new Set(trends.map(t => t.category || 'Unclassified'))).sort();

  if (loading && trends.length === 0) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <Skeleton className="h-8 w-48 bg-slate-800/50" />
          <Skeleton className="h-6 w-32 rounded-full" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div key={i} className="glass-panel p-5 rounded-2xl border border-slate-200 dark:border-slate-800/50">
              {/* Skeleton Content */}
              <div className="flex justify-between mb-4">
                <Skeleton className="h-5 w-20" />
                <Skeleton className="h-8 w-12" />
              </div>
              <Skeleton className="h-6 w-3/4 mb-2" />
              <Skeleton className="h-4 w-full mb-1" />
              <Skeleton className="h-4 w-2/3 mb-4" />
              <div className="flex justify-between mt-4 pt-4 border-t border-slate-100 dark:border-slate-800/50">
                <Skeleton className="h-4 w-24" />
                <Skeleton className="h-4 w-16" />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold tracking-tight mb-1 dark:text-slate-100 text-slate-800">Real-time Stream</h2>
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 bg-cyan-500 dark:bg-cyan-400 rounded-full animate-pulse"></div>
            <span className="text-xs font-mono text-cyan-600 dark:text-cyan-400">LIVE MONITORING</span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Category Pills */}
          <div className="flex items-center gap-2 overflow-x-auto scrollbar-hide mr-4">
            {categories.map(cat => (
              <button
                key={cat}
                onClick={() => onFilterCategory(activeCategory === cat ? null : cat)}
                className={`px-3 py-1.5 rounded-full text-xs font-bold transition-all border ${activeCategory === cat
                  ? 'bg-cyan-600 dark:bg-cyan-500 text-white border-cyan-600 dark:border-cyan-500 shadow-lg shadow-cyan-500/25'
                  : 'bg-white dark:bg-slate-900/50 text-slate-600 dark:text-slate-400 border-slate-200 dark:border-slate-700 hover:border-slate-400 dark:hover:border-slate-500'
                  }`}
              >
                {cat}
              </button>
            ))}
            {activeCategory && (
              <button onClick={() => onFilterCategory(null)} className="p-1 rounded-full hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-500">
                <X size={14} />
              </button>
            )}
          </div>

          {/* Sort Dropdown */}
          <div className="flex items-center bg-white dark:bg-slate-900/50 rounded-lg p-1 border border-slate-200 dark:border-slate-700">
            <button
              onClick={() => onSortBy('score')}
              className={`p-2 rounded-md transition-colors ${sortBy === 'score' ? 'bg-slate-100 dark:bg-slate-800 text-cyan-600 dark:text-cyan-400' : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'}`}
              title="Sort by Impact Score"
              aria-label="Sort by Impact Score"
            >
              <Zap size={16} />
            </button>
            <button
              onClick={() => onSortBy('time')}
              className={`p-2 rounded-md transition-colors ${sortBy === 'time' ? 'bg-slate-100 dark:bg-slate-800 text-cyan-600 dark:text-cyan-400' : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'}`}
              title="Sort by Recency"
              aria-label="Sort by Recency"
            >
              <ArrowUpDown size={16} />
            </button>
          </div>
        </div>
      </div>

      <motion.div
        layout
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
      >
        <AnimatePresence mode="popLayout">
          {trends.map((trend) => (
            <TrendCard key={trend.id} trend={trend} onClick={() => onSelectTrend(trend.id)} />
          ))}
        </AnimatePresence>
      </motion.div>
    </div>
  );
}

function TrendCard({ trend, onClick }: { trend: any, onClick: () => void }) {
  return (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      transition={{ type: "spring", stiffness: 300, damping: 25 }}
      onClick={onClick}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick();
        }
      }}
      role="button"
      tabIndex={0}
      whileHover={{ scale: 1.02, zIndex: 10 }}
      whileFocus={{ scale: 1.02 }}
      className="glass-panel p-5 rounded-2xl group cursor-pointer hover:border-cyan-500/30 transition-all glow-card focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2 dark:focus:ring-offset-slate-900"
    >
      <div className="flex justify-between items-start mb-4">
        <span className="text-xs font-bold bg-slate-200 dark:bg-slate-800 text-slate-600 dark:text-slate-400 px-2 py-1 rounded-md uppercase tracking-wider">
          {trend.category || 'Discovery'}
        </span>
        <div className="flex flex-col items-end">
          <span className="text-cyan-600 dark:text-cyan-400 font-mono text-sm font-bold">{trend.trend_score.toFixed(1)}</span>
          <span className="text-[10px] text-slate-500 uppercase font-black tracking-widest">Score</span>
        </div>
      </div>

      <h3 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-2 leading-tight group-hover:text-cyan-600 dark:group-hover:text-cyan-400 transition-colors">
        {trend.trend_name}
      </h3>

      <p className="text-sm text-slate-600 dark:text-slate-400 line-clamp-2 mb-4">
        {trend.summary || "Pending strategic analysis..."}
      </p>

      <div className="flex items-center justify-between pt-4 border-t border-slate-200 dark:border-slate-800">
        <div className="flex -space-x-2">
          {[1, 2, 3].map(i => (
            <div key={i} className="w-6 h-6 rounded-full bg-slate-200 dark:bg-slate-800 border-2 border-white dark:border-[#020617] flex items-center justify-center text-[10px]">
              👤
            </div>
          ))}
          <span className="pl-4 text-[10px] text-slate-500 font-bold">+{trend.post_count} posts</span>
        </div>
        <span className="text-[10px] text-slate-600 font-mono italic">
          {new Date(trend.last_updated).toLocaleTimeString()}
        </span>
      </div>

      {/* Sparkline */}
      <div className="h-10 mt-2 opacity-50 group-hover:opacity-100 transition-opacity">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={Array.from({ length: 10 }).map((_, i) => ({ value: trend.trend_score + (Math.random() * 2 - 1) }))}>
            <Line type="monotone" dataKey="value" stroke="#0891b2" strokeWidth={2} dot={false} className="hidden dark:block" />
            {/* Sparkline Color */}
            <Line type="monotone" dataKey="value" stroke="#06b6d4" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}
