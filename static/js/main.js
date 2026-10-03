// Real-time system clock in top navigation bar
function updateClock() {
    const el = document.getElementById('system-time');
    if (el) {
        const now = new Date();
        el.innerText = now.toLocaleTimeString();
    }
}
setInterval(updateClock, 1000);
updateClock();

// Global Toast Notification Helper
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast pointer-events-auto flex items-center gap-3 px-4 py-3 rounded-xl shadow-2xl text-xs font-medium border backdrop-blur-md transition-all duration-300 ${
        type === 'success' ? 'bg-emerald-950/90 text-emerald-200 border-emerald-500/40' :
        type === 'error'   ? 'bg-rose-950/90 text-rose-200 border-rose-500/40' :
        type === 'warning' ? 'bg-amber-950/90 text-amber-200 border-amber-500/40' :
                             'bg-slate-900/90 text-slate-200 border-slate-700/60'
    }`;

    const icon = 
        type === 'success' ? '<i class="fa-solid fa-circle-check text-emerald-400 text-sm"></i>' :
        type === 'error'   ? '<i class="fa-solid fa-circle-exclamation text-rose-400 text-sm"></i>' :
        type === 'warning' ? '<i class="fa-solid fa-triangle-exclamation text-amber-400 text-sm"></i>' :
                             '<i class="fa-solid fa-circle-info text-blue-400 text-sm"></i>';

    toast.innerHTML = `
        ${icon}
        <span class="flex-1">${message}</span>
        <button onclick="this.parentElement.remove()" class="text-slate-400 hover:text-white ml-2 text-xs">
            <i class="fa-solid fa-xmark"></i>
        </button>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(10px)';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
