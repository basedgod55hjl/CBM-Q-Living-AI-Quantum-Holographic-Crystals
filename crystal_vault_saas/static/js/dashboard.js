// ================================================================================
// CRYSTAL VAULT - Dashboard JavaScript
// ================================================================================

const API_URL = '';  // Same origin

// State
let authToken = localStorage.getItem('crystal_vault_token');
let currentUser = null;
let entries = [];
let currentEntryId = null;

// ================================================================================
// INITIALIZATION
// ================================================================================

document.addEventListener('DOMContentLoaded', () => {
    if (authToken) {
        checkAuth();
    }
});

async function checkAuth() {
    try {
        const response = await fetch(`${API_URL}/api/auth/me`, {
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
        
        if (response.ok) {
            currentUser = await response.json();
            showDashboard();
            loadEntries();
            loadStats();
        } else {
            logout();
        }
    } catch (error) {
        console.error('Auth check failed:', error);
        logout();
    }
}

// ================================================================================
// AUTH FUNCTIONS
// ================================================================================

function showLogin() {
    document.getElementById('login-form').style.display = 'block';
    document.getElementById('register-form').style.display = 'none';
}

function showRegister() {
    document.getElementById('login-form').style.display = 'none';
    document.getElementById('register-form').style.display = 'block';
}

async function handleLogin(event) {
    event.preventDefault();
    
    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;
    
    try {
        const response = await fetch(`${API_URL}/api/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });
        
        if (response.ok) {
            const data = await response.json();
            authToken = data.access_token;
            currentUser = data.user;
            localStorage.setItem('crystal_vault_token', authToken);
            showDashboard();
            loadEntries();
            loadStats();
        } else {
            const error = await response.json();
            alert(error.detail || 'Login failed');
        }
    } catch (error) {
        console.error('Login error:', error);
        alert('Login failed. Please try again.');
    }
}

async function handleRegister(event) {
    event.preventDefault();
    
    const name = document.getElementById('register-name').value;
    const email = document.getElementById('register-email').value;
    const password = document.getElementById('register-password').value;
    
    try {
        const response = await fetch(`${API_URL}/api/auth/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, email, password })
        });
        
        if (response.ok) {
            const data = await response.json();
            authToken = data.access_token;
            currentUser = data.user;
            localStorage.setItem('crystal_vault_token', authToken);
            showDashboard();
            loadEntries();
            loadStats();
        } else {
            const error = await response.json();
            alert(error.detail || 'Registration failed');
        }
    } catch (error) {
        console.error('Register error:', error);
        alert('Registration failed. Please try again.');
    }
}

async function handleLogout() {
    try {
        await fetch(`${API_URL}/api/auth/logout`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
    } catch (error) {
        console.error('Logout error:', error);
    }
    logout();
}

function logout() {
    authToken = null;
    currentUser = null;
    localStorage.removeItem('crystal_vault_token');
    document.getElementById('auth-screen').style.display = 'flex';
    document.getElementById('dashboard').style.display = 'none';
}

function showDashboard() {
    document.getElementById('auth-screen').style.display = 'none';
    document.getElementById('dashboard').style.display = 'flex';
    
    if (currentUser) {
        document.getElementById('user-name').textContent = currentUser.name;
        document.getElementById('user-tier').textContent = currentUser.tier.toUpperCase();
        document.getElementById('crystal-dna').textContent = currentUser.crystal_dna;
        document.getElementById('settings-email').textContent = currentUser.email;
        document.getElementById('settings-tier').textContent = currentUser.tier;
    }
}

// ================================================================================
// PASSWORD STRENGTH
// ================================================================================

async function checkPasswordStrength(password) {
    if (password.length < 4) {
        document.getElementById('password-strength').innerHTML = '';
        return;
    }
    
    try {
        const response = await fetch(`${API_URL}/api/tools/strength?password=${encodeURIComponent(password)}`, {
            method: 'POST'
        });
        
        if (response.ok) {
            const data = await response.json();
            const colors = {
                'CRYSTAL': '#22c55e',
                'EXCELLENT': '#22c55e',
                'STRONG': '#84cc16',
                'MODERATE': '#f59e0b',
                'WEAK': '#ef4444'
            };
            
            document.getElementById('password-strength').innerHTML = `
                <span style="color: ${colors[data.strength] || '#fff'}">${data.strength}</span>
                - ${data.entropy_bits} bits entropy
            `;
        }
    } catch (error) {
        console.error('Strength check error:', error);
    }
}

// ================================================================================
// ENTRIES
// ================================================================================

async function loadEntries() {
    try {
        const response = await fetch(`${API_URL}/api/vault/entries`, {
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
        
        if (response.ok) {
            const data = await response.json();
            entries = data.entries;
            renderEntries(entries);
        }
    } catch (error) {
        console.error('Load entries error:', error);
    }
}

function renderEntries(entriesToRender) {
    const container = document.getElementById('entries-list');
    
    if (entriesToRender.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="icon">🔐</div>
                <h3>No passwords yet</h3>
                <p>Add your first password to get started</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = entriesToRender.map(entry => `
        <div class="entry-card" onclick="viewEntry('${entry.id}')">
            <div class="entry-icon">${getEntryIcon(entry.category)}</div>
            <div class="entry-details">
                <div class="entry-name">${escapeHtml(entry.name)}</div>
                <div class="entry-username">${escapeHtml(entry.username)}</div>
            </div>
            <span class="entry-category">${entry.category}</span>
        </div>
    `).join('');
}

function getEntryIcon(category) {
    const icons = {
        'general': '🔑',
        'social': '💬',
        'work': '💼',
        'finance': '💰',
        'shopping': '🛒'
    };
    return icons[category] || '🔑';
}

function searchPasswords(query) {
    if (!query) {
        renderEntries(entries);
        return;
    }
    
    const filtered = entries.filter(e => 
        e.name.toLowerCase().includes(query.toLowerCase()) ||
        e.username.toLowerCase().includes(query.toLowerCase()) ||
        e.category.toLowerCase().includes(query.toLowerCase())
    );
    
    renderEntries(filtered);
}

// ================================================================================
// ADD/EDIT ENTRY
// ================================================================================

function showAddModal() {
    document.getElementById('modal-title').textContent = 'Add Password';
    document.getElementById('entry-form').reset();
    document.getElementById('entry-id').value = '';
    document.getElementById('entry-modal').style.display = 'flex';
}

function closeModal() {
    document.getElementById('entry-modal').style.display = 'none';
}

async function handleEntrySubmit(event) {
    event.preventDefault();
    
    const entryId = document.getElementById('entry-id').value;
    const entryData = {
        name: document.getElementById('entry-name').value,
        username: document.getElementById('entry-username').value,
        password: document.getElementById('entry-password').value,
        url: document.getElementById('entry-url').value,
        notes: document.getElementById('entry-notes').value,
        category: document.getElementById('entry-category').value
    };
    
    try {
        const url = entryId 
            ? `${API_URL}/api/vault/entries/${entryId}`
            : `${API_URL}/api/vault/entries`;
        
        const response = await fetch(url, {
            method: entryId ? 'PUT' : 'POST',
            headers: {
                'Authorization': `Bearer ${authToken}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(entryData)
        });
        
        if (response.ok) {
            closeModal();
            loadEntries();
            loadStats();
        } else {
            const error = await response.json();
            alert(error.detail || 'Failed to save entry');
        }
    } catch (error) {
        console.error('Save entry error:', error);
        alert('Failed to save entry');
    }
}

function togglePasswordVisibility() {
    const input = document.getElementById('entry-password');
    input.type = input.type === 'password' ? 'text' : 'password';
}

async function generateForEntry() {
    try {
        const response = await fetch(`${API_URL}/api/tools/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ length: 20 })
        });
        
        if (response.ok) {
            const data = await response.json();
            document.getElementById('entry-password').value = data.password;
            document.getElementById('entry-password').type = 'text';
        }
    } catch (error) {
        console.error('Generate error:', error);
    }
}

// ================================================================================
// VIEW ENTRY
// ================================================================================

async function viewEntry(entryId) {
    currentEntryId = entryId;
    
    try {
        const response = await fetch(`${API_URL}/api/vault/entries/${entryId}`, {
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
        
        if (response.ok) {
            const entry = await response.json();
            
            document.getElementById('view-title').textContent = entry.name;
            document.getElementById('view-username').textContent = entry.username;
            document.getElementById('view-password').textContent = entry.password;
            document.getElementById('view-password').dataset.password = entry.password;
            document.getElementById('view-password').classList.add('password-hidden');
            document.getElementById('view-password').textContent = '••••••••';
            document.getElementById('view-url').textContent = entry.url || '-';
            document.getElementById('view-url').href = entry.url || '#';
            document.getElementById('view-notes').textContent = entry.notes || '-';
            
            document.getElementById('view-modal').style.display = 'flex';
        }
    } catch (error) {
        console.error('View entry error:', error);
    }
}

function closeViewModal() {
    document.getElementById('view-modal').style.display = 'none';
    currentEntryId = null;
}

function toggleViewPassword() {
    const el = document.getElementById('view-password');
    if (el.classList.contains('password-hidden')) {
        el.textContent = el.dataset.password;
        el.classList.remove('password-hidden');
    } else {
        el.textContent = '••••••••';
        el.classList.add('password-hidden');
    }
}

async function editEntry() {
    closeViewModal();
    
    const entry = entries.find(e => e.id === currentEntryId);
    if (!entry) return;
    
    // Fetch full entry with password
    try {
        const response = await fetch(`${API_URL}/api/vault/entries/${currentEntryId}`, {
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
        
        if (response.ok) {
            const fullEntry = await response.json();
            
            document.getElementById('modal-title').textContent = 'Edit Password';
            document.getElementById('entry-id').value = currentEntryId;
            document.getElementById('entry-name').value = fullEntry.name;
            document.getElementById('entry-username').value = fullEntry.username;
            document.getElementById('entry-password').value = fullEntry.password;
            document.getElementById('entry-url').value = fullEntry.url || '';
            document.getElementById('entry-notes').value = fullEntry.notes || '';
            document.getElementById('entry-category').value = fullEntry.category;
            
            document.getElementById('entry-modal').style.display = 'flex';
        }
    } catch (error) {
        console.error('Edit entry error:', error);
    }
}

async function deleteEntry() {
    if (!confirm('Are you sure you want to delete this password?')) return;
    
    try {
        const response = await fetch(`${API_URL}/api/vault/entries/${currentEntryId}`, {
            method: 'DELETE',
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
        
        if (response.ok) {
            closeViewModal();
            loadEntries();
            loadStats();
        }
    } catch (error) {
        console.error('Delete entry error:', error);
    }
}

function copyField(fieldId) {
    const el = document.getElementById(fieldId);
    const text = el.dataset.password || el.textContent;
    navigator.clipboard.writeText(text);
    
    // Visual feedback
    const originalText = el.textContent;
    el.textContent = 'Copied!';
    setTimeout(() => {
        el.textContent = originalText;
    }, 1000);
}

// ================================================================================
// GENERATOR
// ================================================================================

function updateLengthValue(value) {
    document.getElementById('length-value').textContent = value;
}

function updateWordsValue(value) {
    document.getElementById('words-value').textContent = value;
}

async function generatePassword() {
    const params = {
        length: parseInt(document.getElementById('password-length').value),
        use_lowercase: document.getElementById('use-lowercase').checked,
        use_uppercase: document.getElementById('use-uppercase').checked,
        use_digits: document.getElementById('use-digits').checked,
        use_symbols: document.getElementById('use-symbols').checked
    };
    
    try {
        const response = await fetch(`${API_URL}/api/tools/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        
        if (response.ok) {
            const data = await response.json();
            document.getElementById('generated-password').value = data.password;
            updateStrengthDisplay(data.strength);
        }
    } catch (error) {
        console.error('Generate error:', error);
    }
}

async function generatePassphrase() {
    const wordCount = parseInt(document.getElementById('word-count').value);
    
    try {
        const response = await fetch(`${API_URL}/api/tools/passphrase`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word_count: wordCount })
        });
        
        if (response.ok) {
            const data = await response.json();
            document.getElementById('generated-passphrase').value = data.passphrase;
        }
    } catch (error) {
        console.error('Generate passphrase error:', error);
    }
}

function updateStrengthDisplay(strength) {
    const fill = document.querySelector('.strength-fill');
    const text = document.querySelector('.strength-text');
    
    const colors = {
        'CRYSTAL': '#22c55e',
        'EXCELLENT': '#22c55e',
        'STRONG': '#84cc16',
        'MODERATE': '#f59e0b',
        'WEAK': '#ef4444'
    };
    
    fill.style.width = `${strength.score}%`;
    fill.style.background = colors[strength.strength] || '#fff';
    text.textContent = `${strength.strength} - ${strength.entropy_bits} bits - ${strength.estimated_crack_time}`;
}

function copyPassword() {
    const password = document.getElementById('generated-password').value;
    navigator.clipboard.writeText(password);
    alert('Password copied!');
}

function copyPassphrase() {
    const passphrase = document.getElementById('generated-passphrase').value;
    navigator.clipboard.writeText(passphrase);
    alert('Passphrase copied!');
}

// ================================================================================
// STATS
// ================================================================================

async function loadStats() {
    try {
        const response = await fetch(`${API_URL}/api/vault/stats`, {
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
        
        if (response.ok) {
            const stats = await response.json();
            document.getElementById('entry-count').textContent = `${stats.total_entries} entries`;
            document.getElementById('tier-info').textContent = `${stats.tier} tier`;
        }
    } catch (error) {
        console.error('Load stats error:', error);
    }
}

// ================================================================================
// NAVIGATION
// ================================================================================

function showSection(sectionName) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(s => s.style.display = 'none');
    
    // Show selected section
    document.getElementById(`${sectionName}-section`).style.display = 'block';
    
    // Update nav
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    event.target.closest('.nav-item').classList.add('active');
}

// ================================================================================
// SETTINGS
// ================================================================================

function upgradeTier() {
    alert('Upgrade feature coming soon! Contact support for enterprise pricing.');
}

function exportVault() {
    alert('Export feature coming soon!');
}

function deleteAccount() {
    if (confirm('Are you sure? This will permanently delete all your data.')) {
        alert('Account deletion coming soon. Contact support.');
    }
}

// ================================================================================
// UTILITIES
// ================================================================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

