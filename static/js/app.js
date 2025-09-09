// Global JavaScript for BTC Options Trader UI

// Global variables
let statusUpdateInterval = null;
let notifications = [];

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    startStatusUpdates();
});

function initializeApp() {
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        document.body.setAttribute('data-theme', savedTheme);
    }
    
    // Initialize tooltips
    initializeTooltips();
    
    // Update bot status badge in navbar
    updateBotStatusBadge();
    
    console.log('BTC Options Trader UI initialized');
}

function setupEventListeners() {
    // Global error handling
    window.addEventListener('error', function(e) {
        console.error('Global error:', e.error);
        showNotification('error', 'An error occurred: ' + e.message);
    });
    
    // Handle form submissions
    document.addEventListener('submit', function(e) {
        if (e.target.classList.contains('needs-validation')) {
            if (!e.target.checkValidity()) {
                e.preventDefault();
                e.stopPropagation();
            }
            e.target.classList.add('was-validated');
        }
    });
    
    // Handle API test buttons globally
    document.addEventListener('click', function(e) {
        if (e.target.matches('[data-action="test-api"]')) {
            testApiConnection(e.target);
        }
    });
}

function initializeTooltips() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Status updates
function startStatusUpdates() {
    updateBotStatus();
    statusUpdateInterval = setInterval(updateBotStatus, 10000); // Every 10 seconds
}

function updateBotStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            updateStatusDisplay(data);
            updateBotStatusBadge(data.status);
        })
        .catch(error => {
            console.error('Error updating status:', error);
            updateBotStatusBadge('Error');
        });
}

function updateStatusDisplay(data) {
    // Update various status elements if they exist on the page
    const elements = {
        'daily-pnl': data.daily_pnl ? `$${data.daily_pnl.toFixed(2)}` : '$0.00',
        'current-positions': data.current_positions || 0,
        'total-trades': data.total_trades || 0,
        'last-update': data.last_update ? formatDateTime(data.last_update) : 'Never'
    };
    
    Object.entries(elements).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
            
            // Add color classes for P&L
            if (id === 'daily-pnl' && data.daily_pnl !== undefined) {
                element.className = data.daily_pnl >= 0 ? 'text-success' : 'text-danger';
            }
        }
    });
    
    // Update bot status display
    const statusElement = document.getElementById('status-display');
    if (statusElement) {
        statusElement.textContent = data.status || 'Unknown';
        statusElement.className = data.status === 'Running' ? 'text-success' : 'text-danger';
    }
    
    // Update control buttons
    updateControlButtons(data.status === 'Running');
}

function updateBotStatusBadge(status = 'Checking...') {
    const badge = document.getElementById('bot-status');
    if (badge) {
        badge.textContent = status;
        
        const statusClasses = {
            'Running': 'bg-success',
            'Stopped': 'bg-secondary',
            'Starting...': 'bg-warning',
            'Stopping...': 'bg-warning',
            'Error': 'bg-danger'
        };
        
        // Remove all status classes
        Object.values(statusClasses).forEach(cls => badge.classList.remove(cls));
        
        // Add appropriate class
        const newClass = statusClasses[status] || 'bg-secondary';
        badge.classList.add(newClass);
    }
}

function updateControlButtons(isRunning) {
    const startBtn = document.getElementById('start-bot');
    const stopBtn = document.getElementById('stop-bot');
    
    if (startBtn) {
        startBtn.disabled = isRunning;
    }
    
    if (stopBtn) {
        stopBtn.disabled = !isRunning;
    }
}

// API functions
function testApiConnection(button) {
    const originalText = button.innerHTML;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Testing...';
    button.disabled = true;
    
    const formData = new FormData();
    const apiKeyElement = document.getElementById('api_key');
    const apiSecretElement = document.getElementById('api_secret');
    const paperTradingElement = document.getElementById('paper_trading');
    
    if (apiKeyElement) formData.append('api_key', apiKeyElement.value);
    if (apiSecretElement) formData.append('api_secret', apiSecretElement.value);
    if (paperTradingElement) formData.append('paper_trading', paperTradingElement.checked ? 'on' : '');
    
    fetch('/test_api', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const resultElement = document.getElementById('api-test-result');
        if (resultElement) {
            resultElement.innerHTML = createAlert(
                data.success ? 'success' : 'danger',
                data.message
            );
        } else {
            showNotification(
                data.success ? 'success' : 'error',
                data.message
            );
        }
    })
    .catch(error => {
        showNotification('error', 'Test failed: ' + error.message);
    })
    .finally(() => {
        button.innerHTML = originalText;
        button.disabled = false;
    });
}

function startBot() {
    return fetch('/start_bot', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            showNotification(
                data.success ? 'success' : 'error',
                data.message
            );
            if (data.success) {
                updateBotStatus(); // Immediate status update
            }
            return data;
        });
}

function stopBot() {
    return fetch('/stop_bot', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            showNotification(
                data.success ? 'success' : 'error',
                data.message
            );
            if (data.success) {
                updateBotStatus(); // Immediate status update
            }
            return data;
        });
}

// UI Utility functions
function showNotification(type, message, duration = 5000) {
    const alertClass = type === 'error' ? 'alert-danger' : `alert-${type}`;
    const iconClass = {
        'success': 'fas fa-check-circle',
        'error': 'fas fa-exclamation-circle',
        'warning': 'fas fa-exclamation-triangle',
        'info': 'fas fa-info-circle'
    }[type] || 'fas fa-info-circle';
    
    const alertHTML = `
        <div class="alert ${alertClass} alert-dismissible fade show" role="alert">
            <i class="${iconClass}"></i> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    // Find a container for the alert
    let container = document.querySelector('.alert-container');
    if (!container) {
        container = document.querySelector('.container-fluid');
        if (!container) {
            container = document.body;
        }
    }
    
    const alertDiv = document.createElement('div');
    alertDiv.innerHTML = alertHTML;
    const alertElement = alertDiv.firstElementChild;
    
    // Insert at the top of the container
    container.insertBefore(alertElement, container.firstChild);
    
    // Auto-remove after duration
    setTimeout(() => {
        if (alertElement && alertElement.parentNode) {
            alertElement.remove();
        }
    }, duration);
    
    return alertElement;
}

function createAlert(type, message, dismissible = true) {
    const iconClass = {
        'success': 'fas fa-check-circle',
        'danger': 'fas fa-exclamation-circle',
        'warning': 'fas fa-exclamation-triangle',
        'info': 'fas fa-info-circle'
    }[type] || 'fas fa-info-circle';
    
    const dismissibleClass = dismissible ? 'alert-dismissible' : '';
    const dismissButton = dismissible ? '<button type="button" class="btn-close" data-bs-dismiss="alert"></button>' : '';
    
    return `
        <div class="alert alert-${type} ${dismissibleClass} fade show" role="alert">
            <i class="${iconClass}"></i> ${message}
            ${dismissButton}
        </div>
    `;
}

function formatDateTime(isoString) {
    try {
        const date = new Date(isoString);
        return date.toLocaleString();
    } catch (e) {
        return isoString;
    }
}

function formatCurrency(amount, currency = 'USD') {
    try {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency,
            minimumFractionDigits: 2,
            maximumFractionDigits: 6
        }).format(amount);
    } catch (e) {
        return `$${amount.toFixed(2)}`;
    }
}

function formatPercentage(value, decimals = 1) {
    return `${value.toFixed(decimals)}%`;
}

// Form validation helpers
function validateApiCredentials() {
    const apiKey = document.getElementById('api_key')?.value.trim();
    const apiSecret = document.getElementById('api_secret')?.value.trim();
    
    if (!apiKey || !apiSecret) {
        showNotification('error', 'Both API Key and Secret are required');
        return false;
    }
    
    if (apiKey.length < 10 || apiSecret.length < 10) {
        showNotification('error', 'API credentials appear to be invalid (too short)');
        return false;
    }
    
    return true;
}

function validateRiskParameters() {
    const portfolioSize = parseFloat(document.getElementById('portfolio_size')?.value) || 0;
    const positionSize = parseFloat(document.getElementById('position_size_usd')?.value) || 0;
    const maxDailyLoss = parseFloat(document.getElementById('max_daily_loss')?.value) || 0;
    
    if (positionSize > portfolioSize * 0.2) {
        const confirm = window.confirm(
            'Position size is more than 20% of portfolio. This is very high risk. Continue?'
        );
        if (!confirm) return false;
    }
    
    if (maxDailyLoss > portfolioSize * 0.1) {
        const confirm = window.confirm(
            'Max daily loss is more than 10% of portfolio. This is high risk. Continue?'
        );
        if (!confirm) return false;
    }
    
    return true;
}

// Theme management
function toggleTheme() {
    const currentTheme = document.body.getAttribute('data-theme') || 'light';
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    
    document.body.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    
    // Update theme toggle button if it exists
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.innerHTML = newTheme === 'light' ? 
            '<i class="fas fa-moon"></i>' : 
            '<i class="fas fa-sun"></i>';
    }
}

// Chart utilities (if Chart.js is available)
function createPnLChart(canvasId, data) {
    if (typeof Chart === 'undefined') {
        console.warn('Chart.js not available');
        return null;
    }
    
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(d => d.date),
            datasets: [{
                label: 'Cumulative P&L',
                data: data.map(d => d.pnl),
                borderColor: 'rgb(0, 123, 255)',
                backgroundColor: 'rgba(0, 123, 255, 0.1)',
                tension: 0.2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'P&L ($)'
                    }
                }
            }
        }
    });
}

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (statusUpdateInterval) {
        clearInterval(statusUpdateInterval);
    }
});

// Export functions for global access
window.BTCTrader = {
    startBot,
    stopBot,
    testApiConnection,
    showNotification,
    updateBotStatus,
    validateApiCredentials,
    validateRiskParameters,
    formatCurrency,
    formatPercentage,
    formatDateTime,
    toggleTheme
};