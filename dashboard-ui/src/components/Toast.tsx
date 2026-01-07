"use client";

import React, { createContext, useContext, useState, ReactNode } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { CheckCircle, AlertTriangle, AlertCircle, Info, X } from "lucide-react";

type ToastType = "success" | "error" | "warning" | "info";

interface Toast {
    id: string;
    message: string;
    type: ToastType;
}

interface ToastContextValue {
    showToast: (message: string, type?: ToastType) => void;
}

const ToastContext = createContext<ToastContextValue | undefined>(undefined);

export function useToast() {
    const context = useContext(ToastContext);
    if (!context) {
        throw new Error("useToast must be used within a ToastProvider");
    }
    return context;
}

export function ToastProvider({ children }: { children: ReactNode }) {
    const [toasts, setToasts] = useState<Toast[]>([]);

    const showToast = (message: string, type: ToastType = "info") => {
        const id = Date.now().toString(36) + Math.random().toString(36).substr(2);
        const newToast = { id, message, type };
        setToasts((prev) => [...prev, newToast]);

        // Auto dismiss
        setTimeout(() => {
            removeToast(id);
        }, 4000);
    };

    const removeToast = (id: string) => {
        setToasts((prev) => prev.filter((t) => t.id !== id));
    };

    return (
        <ToastContext.Provider value={{ showToast }}>
            {children}
            <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2">
                <AnimatePresence>
                    {toasts.map((toast) => (
                        <ToastItem key={toast.id} toast={toast} onClose={() => removeToast(toast.id)} />
                    ))}
                </AnimatePresence>
            </div>
        </ToastContext.Provider>
    );
}

function ToastItem({ toast, onClose }: { toast: Toast; onClose: () => void }) {
    const icons = {
        success: <CheckCircle size={18} className="text-emerald-400" />,
        error: <AlertCircle size={18} className="text-red-400" />,
        warning: <AlertTriangle size={18} className="text-amber-400" />,
        info: <Info size={18} className="text-blue-400" />,
    };

    const bgColors = {
        success: "bg-emerald-500/10 border-emerald-500/20",
        error: "bg-red-500/10 border-red-500/20",
        warning: "bg-amber-500/10 border-amber-500/20",
        info: "bg-blue-500/10 border-blue-500/20",
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9, transition: { duration: 0.2 } }}
            className={`min-w-[300px] p-4 rounded-xl border backdrop-blur-md shadow-lg flex items-start gap-3 relative overflow-hidden ${bgColors[toast.type]}`}
        >
            <div className="mt-0.5">{icons[toast.type]}</div>
            <p className="text-sm font-medium text-slate-200 pr-4">{toast.message}</p>
            <button
                onClick={onClose}
                className="absolute top-2 right-2 text-slate-500 hover:text-slate-300 transition-colors"
            >
                <X size={14} />
            </button>
        </motion.div>
    );
}
