// Claude Shell App - Frontend JavaScript

const API = '/api';
let currentProject = null;

// DOM Elements
const projectList = document.getElementById('project-list');
const projectForm = document.getElementById('project-form');
const mainContent = document.getElementById('main-content');
const emptyState = document.getElementById('empty-state');
const projectView = document.getElementById('project-view');
const projectTitle = document.getElementById('project-title');
const tabs = document.querySelectorAll('.tab');
const sections = document.querySelectorAll('.section');

// Tab switching
tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        const target = tab.dataset.tab;
        tabs.forEach(t => t.classList.remove('active'));
        sections.forEach(s => s.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(`section-${target}`).classList.add('active');
    });
});

// Load projects list
async function loadProjects() {
    const res = await fetch(`${API}/projects`);
    const projects = await res.json();

    projectList.innerHTML = '';
    projects.forEach(p => {
        const li = document.createElement('li');
        li.dataset.id = p.id;
        li.innerHTML = `
            <div class="name">${escapeHtml(p.name)}</div>
            <div class="desc">${escapeHtml(p.description || 'No description')}</div>
        `;
        li.addEventListener('click', () => selectProject(p.id));
        if (currentProject && currentProject.id === p.id) {
            li.classList.add('active');
        }
        projectList.appendChild(li);
    });
}

// Create new project
projectForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(projectForm);

    const res = await fetch(`${API}/projects`, {
        method: 'POST',
        body: formData
    });

    if (res.ok) {
        const project = await res.json();
        projectForm.reset();
        await loadProjects();
        selectProject(project.id);
    }
});

// Select and load a project
async function selectProject(projectId) {
    const res = await fetch(`${API}/projects/${projectId}`);
    if (!res.ok) return;

    currentProject = await res.json();

    // Update UI
    emptyState.style.display = 'none';
    projectView.style.display = 'block';
    projectTitle.textContent = currentProject.name;

    // Update active state in list
    document.querySelectorAll('#project-list li').forEach(li => {
        li.classList.toggle('active', li.dataset.id === projectId);
    });

    // Load all sections
    loadNotes();
    loadPins();
    loadUploads();
    loadLog();
}

// Notes
function loadNotes() {
    document.getElementById('notes-editor').value = currentProject.notes || '';
}

document.getElementById('notes-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const notes = document.getElementById('notes-editor').value;

    const formData = new FormData();
    formData.append('notes', notes);

    const res = await fetch(`${API}/projects/${currentProject.id}/notes`, {
        method: 'POST',
        body: formData
    });

    if (res.ok) {
        showStatus('notes-status', 'Notes saved!', 'success');
        currentProject.notes = notes;
        loadLog();
    }
});

// Pins
function loadPins() {
    const container = document.getElementById('pins-list');
    const pins = currentProject.pins || [];

    if (pins.length === 0) {
        container.innerHTML = '<p style="color: var(--text-muted)">No pinned prompts yet.</p>';
        return;
    }

    container.innerHTML = pins.map(pin => `
        <div class="pin-card" data-id="${pin.id}">
            <button class="delete-btn" onclick="deletePin('${pin.id}')">&times;</button>
            <h4>${escapeHtml(pin.title)}</h4>
            <pre>${escapeHtml(pin.content)}</pre>
        </div>
    `).join('');
}

document.getElementById('pin-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);

    const res = await fetch(`${API}/projects/${currentProject.id}/pin`, {
        method: 'POST',
        body: formData
    });

    if (res.ok) {
        const pin = await res.json();
        currentProject.pins = currentProject.pins || [];
        currentProject.pins.push(pin);
        loadPins();
        e.target.reset();
        loadLog();
    }
});

async function deletePin(pinId) {
    const res = await fetch(`${API}/projects/${currentProject.id}/pin/${pinId}`, {
        method: 'DELETE'
    });

    if (res.ok) {
        currentProject.pins = currentProject.pins.filter(p => p.id !== pinId);
        loadPins();
        loadLog();
    }
}

// Uploads
function loadUploads() {
    const container = document.getElementById('uploads-list');
    const uploads = currentProject.uploads || [];

    if (uploads.length === 0) {
        container.innerHTML = '<p style="color: var(--text-muted)">No files uploaded yet.</p>';
        return;
    }

    container.innerHTML = uploads.map(u => `
        <div class="upload-item">
            <div>
                <div class="name">${escapeHtml(u.label || u.original_name)}</div>
                <div class="meta">${escapeHtml(u.original_name)} - ${formatSize(u.size)}</div>
            </div>
        </div>
    `).join('');
}

document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);

    const res = await fetch(`${API}/projects/${currentProject.id}/upload`, {
        method: 'POST',
        body: formData
    });

    if (res.ok) {
        const upload = await res.json();
        currentProject.uploads = currentProject.uploads || [];
        currentProject.uploads.push(upload);
        loadUploads();
        e.target.reset();
        showStatus('upload-status', 'File uploaded!', 'success');
        loadLog();
    }
});

// Log
function loadLog() {
    const container = document.getElementById('log-list');
    const log = currentProject.log || [];

    if (log.length === 0) {
        container.innerHTML = '<p style="color: var(--text-muted)">No events logged yet.</p>';
        return;
    }

    // Show newest first
    container.innerHTML = [...log].reverse().map(entry => `
        <div class="log-entry">
            <span class="time">${formatTime(entry.timestamp)}</span>
            <span class="kind">${escapeHtml(entry.kind)}</span>
            <span class="message">${escapeHtml(entry.message)}</span>
        </div>
    `).join('');
}

// Export
document.getElementById('export-btn').addEventListener('click', async () => {
    if (!currentProject) return;
    window.open(`${API}/projects/${currentProject.id}/export`, '_blank');
});

// Utility functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function formatTime(isoString) {
    const date = new Date(isoString);
    return date.toLocaleString();
}

function showStatus(elementId, message, type) {
    const el = document.getElementById(elementId);
    el.textContent = message;
    el.className = `status ${type}`;
    setTimeout(() => {
        el.className = 'status';
    }, 3000);
}

// Initialize
loadProjects();
