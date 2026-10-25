/**
 * MedExplain AI — Frontend Application Logic
 * Ultra-Luxury 3D Visualization, Conversational Intelligence & Agentic Loop
 */

document.addEventListener('DOMContentLoaded', () => {

    // ─── STATE MANAGEMENT ───
    const state = {
        selectedFile: null,
        pastedText: '',
        activeTab: 'upload', // 'upload', 'paste', 'samples'
        activeReportText: '',
        activeFlags: [],
        activeCritical: [],
        chatHistory: [],
        isAnalyzing: false,
        isChatting: false,
        speechRecognition: null,
        isRecording: false
    };

    // ─── SAMPLE DATA PRESETS ───
    const SAMPLE_REPORTS = {
        anemia: `PATIENT COMPLETE BLOOD COUNT (CBC)
Hemoglobin: 9.2 g/dL (Ref: 13.0 - 17.0) [LOW]
Hematocrit: 28.5 % (Ref: 38.0 - 50.0) [LOW]
Red Blood Cells (RBC): 3.2 10^6/uL (Ref: 4.5 - 5.9) [LOW]
White Blood Cells (WBC): 5800 /uL (Ref: 4000 - 11000) [NORMAL]
Platelet Count: 210000 /uL (Ref: 150000 - 450000) [NORMAL]
Serum Ferritin: 8.0 ng/mL (Ref: 20 - 250) [LOW]`,

        critical: `EMERGENCY METABOLIC PANEL
Potassium: 6.8 mmol/L (Ref: 3.5 - 5.0) [CRITICAL HIGH]
Sodium: 138.0 mmol/L (Ref: 135.0 - 145.0) [NORMAL]
Serum Creatinine: 1.8 mg/dL (Ref: 0.7 - 1.3) [HIGH]
Blood Glucose: 105.0 mg/dL (Ref: 70 - 99) [HIGH]`,

        lipid: `FASTING LIPID PANEL
Total Cholesterol: 245.0 mg/dL (Ref: 125.0 - 200.0) [HIGH]
LDL Cholesterol: 162.0 mg/dL (Ref: 50.0 - 100.0) [HIGH]
HDL Cholesterol: 38.0 mg/dL (Ref: 40.0 - 60.0) [LOW]
Triglycerides: 220.0 mg/dL (Ref: 50.0 - 150.0) [HIGH]`,

        normal: `ROUTINE HEALTH CHECK COMPLETE BLOOD COUNT
Hemoglobin: 14.8 g/dL (Ref: 13.0 - 17.0) [NORMAL]
White Blood Cells (WBC): 6500 /uL (Ref: 4000 - 11000) [NORMAL]
Red Blood Cells (RBC): 4.9 10^6/uL (Ref: 4.5 - 5.9) [NORMAL]
Platelet Count: 260000 /uL (Ref: 150000 - 450000) [NORMAL]
Hematocrit: 44.0 % (Ref: 38.0 - 50.0) [NORMAL]`
    };

    // ─── DOM ELEMENTS ───
    const promptInput = document.getElementById('promptInput');
    const sendBtn = document.getElementById('sendBtn');
    const micBtn = document.getElementById('micBtn');
    const promptAttachBtn = document.getElementById('promptAttachBtn');
    const attachDropdown = document.getElementById('attachDropdown');
    const printReportBtn = document.getElementById('printReportBtn');

    // Analyzer Elements
    const tabUploadBtn = document.getElementById('tabUploadBtn');
    const tabPasteBtn = document.getElementById('tabPasteBtn');
    const tabSamplesBtn = document.getElementById('tabSamplesBtn');
    const paneUpload = document.getElementById('paneUpload');
    const panePaste = document.getElementById('panePaste');
    const paneSamples = document.getElementById('paneSamples');
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const browseFileBtn = document.getElementById('browseFileBtn');
    const selectedFileBadge = document.getElementById('selectedFileBadge');
    const selectedFileName = document.getElementById('selectedFileName');
    const removeFileBtn = document.getElementById('removeFileBtn');
    const pasteTextArea = document.getElementById('pasteTextArea');
    const languageSelect = document.getElementById('languageSelect');
    const analyzeSubmitBtn = document.getElementById('analyzeSubmitBtn');

    // Results Elements
    const resultsSection = document.getElementById('resultsSection');
    const criticalBanner = document.getElementById('criticalBanner');
    const criticalMessage = document.getElementById('criticalMessage');
    const metricTotalTests = document.getElementById('metricTotalTests');
    const metricAbnormalFlags = document.getElementById('metricAbnormalFlags');
    const metricCriticalCount = document.getElementById('metricCriticalCount');
    const metricTokensUsed = document.getElementById('metricTokensUsed');
    const flagsTableBody = document.getElementById('flagsTableBody');
    const explanationText = document.getElementById('explanationText');

    // Chat Elements
    const chatMessagesContainer = document.getElementById('chatMessagesContainer');
    const chatTypingIndicator = document.getElementById('chatTypingIndicator');
    const chatBarInput = document.getElementById('chatBarInput');
    const chatBarSendBtn = document.getElementById('chatBarSendBtn');
    const chatClearBtn = document.getElementById('chatClearBtn');

    // ─── INITIALIZE 3D VISUALIZATION ───
    init3DBiomarkerVisualizer();

    // ─── NAVIGATION SCROLL HANDLERS ───
    document.querySelectorAll('.dock-btn[data-scroll]').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.dock-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const targetId = btn.getAttribute('data-scroll');
            const targetEl = document.getElementById(targetId);
            if (targetEl) {
                targetEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });

    document.getElementById('dockNewChatBtn')?.addEventListener('click', () => {
        document.getElementById('analyzerSection')?.scrollIntoView({ behavior: 'smooth' });
    });

    document.getElementById('cardUploadReport')?.addEventListener('click', () => {
        document.getElementById('analyzerSection')?.scrollIntoView({ behavior: 'smooth' });
    });

    document.getElementById('cardBiomarkerFlags')?.addEventListener('click', () => {
        if (state.activeFlags.length > 0) {
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        } else {
            document.getElementById('analyzerSection')?.scrollIntoView({ behavior: 'smooth' });
        }
    });

    document.getElementById('cardDoctorLifestyle')?.addEventListener('click', () => {
        document.getElementById('chatSection')?.scrollIntoView({ behavior: 'smooth' });
    });

    printReportBtn?.addEventListener('click', () => {
        window.print();
    });

    // ─── ATTACH / SAMPLE MENU TOGGLE ───
    promptAttachBtn?.addEventListener('click', (e) => {
        e.stopPropagation();
        attachDropdown?.classList.toggle('show');
    });

    document.addEventListener('click', () => {
        attachDropdown?.classList.remove('show');
    });

    document.getElementById('attachUploadAction')?.addEventListener('click', () => {
        fileInput?.click();
    });

    document.getElementById('attachSampleAnemia')?.addEventListener('click', () => {
        loadSamplePreset('anemia');
    });

    document.getElementById('attachSamplePotassium')?.addEventListener('click', () => {
        loadSamplePreset('critical');
    });

    document.getElementById('attachSampleLipid')?.addEventListener('click', () => {
        loadSamplePreset('lipid');
    });

    // ─── TABS HANDLING ───
    tabUploadBtn?.addEventListener('click', () => switchTab('upload'));
    tabPasteBtn?.addEventListener('click', () => switchTab('paste'));
    tabSamplesBtn?.addEventListener('click', () => switchTab('samples'));

    function switchTab(tab) {
        state.activeTab = tab;
        tabUploadBtn?.classList.toggle('active', tab === 'upload');
        tabPasteBtn?.classList.toggle('active', tab === 'paste');
        tabSamplesBtn?.classList.toggle('active', tab === 'samples');

        paneUpload?.classList.toggle('active', tab === 'upload');
        panePaste?.classList.toggle('active', tab === 'paste');
        paneSamples?.classList.toggle('active', tab === 'samples');
    }

    // ─── FILE UPLOAD & DROPZONE ───
    browseFileBtn?.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput?.click();
    });

    dropzone?.addEventListener('click', () => {
        fileInput?.click();
    });

    dropzone?.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone?.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone?.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput?.addEventListener('change', () => {
        if (fileInput.files && fileInput.files.length > 0) {
            handleFileSelect(fileInput.files[0]);
        }
    });

    function handleFileSelect(file) {
        state.selectedFile = file;
        if (selectedFileName) selectedFileName.textContent = file.name;
        if (selectedFileBadge) selectedFileBadge.style.display = 'inline-flex';
    }

    removeFileBtn?.addEventListener('click', (e) => {
        e.stopPropagation();
        state.selectedFile = null;
        if (fileInput) fileInput.value = '';
        if (selectedFileBadge) selectedFileBadge.style.display = 'none';
    });

    // ─── SAMPLE CHIP CLICKS ───
    document.querySelectorAll('.sample-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const sampleKey = chip.getAttribute('data-sample');
            loadSamplePreset(sampleKey);
        });
    });

    function loadSamplePreset(key) {
        const text = SAMPLE_REPORTS[key] || '';
        state.pastedText = text;
        if (pasteTextArea) pasteTextArea.value = text;
        switchTab('paste');
        document.getElementById('analyzerSection')?.scrollIntoView({ behavior: 'smooth' });
    }

    // ─── ANALYZE REPORT SUBMIT ───
    analyzeSubmitBtn?.addEventListener('click', runReportAnalysis);

    async function runReportAnalysis() {
        if (state.isAnalyzing) return;

        const language = languageSelect?.value || 'English';
        const formData = new FormData();
        formData.append('language', language);

        if (state.activeTab === 'upload' && state.selectedFile) {
            formData.append('file', state.selectedFile);
        } else {
            const rawText = pasteTextArea?.value.trim() || state.pastedText.trim();
            if (!rawText && !state.selectedFile) {
                alert('Please upload a report file or paste lab text to analyze.');
                return;
            }
            formData.append('text', rawText);
        }

        try {
            state.isAnalyzing = true;
            analyzeSubmitBtn.disabled = true;
            analyzeSubmitBtn.innerHTML = `<span>Analyzing Lab Report...</span>`;

            const resp = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                throw new Error(err.detail || `Server error: ${resp.status}`);
            }

            const data = await resp.json();
            displayAnalysisResults(data);

        } catch (error) {
            alert(`Analysis failed: ${error.message}`);
        } finally {
            state.isAnalyzing = false;
            analyzeSubmitBtn.disabled = false;
            analyzeSubmitBtn.innerHTML = `<span>Analyze Report</span>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="5" y1="12" x2="19" y2="12"></line>
                    <polyline points="12 5 19 12 12 19"></polyline>
                </svg>`;
        }
    }

    function displayAnalysisResults(data) {
        // Save state for chat
        state.activeFlags = data.flags || [];
        state.activeCritical = data.critical_findings || [];
        state.activeReportText = pasteTextArea?.value || '';

        // Reveal results section
        if (resultsSection) resultsSection.style.display = 'flex';

        // Critical Alert Banner
        if (data.critical_findings && data.critical_findings.length > 0) {
            if (criticalBanner) criticalBanner.style.display = 'flex';
            if (criticalMessage) {
                const names = data.critical_findings.map(c => c.test.toUpperCase()).join(', ');
                criticalMessage.innerHTML = `<strong>Attention:</strong> Severe / panic values detected in <strong>${names}</strong>. Please contact your doctor or hospital today.`;
            }
        } else {
            if (criticalBanner) criticalBanner.style.display = 'none';
        }

        // Metrics Counters
        const total = data.flags ? data.flags.length : 0;
        const abnormal = data.flags ? data.flags.filter(f => f.flag === 'HIGH' || f.flag === 'LOW').length : 0;
        const critCount = data.critical_findings ? data.critical_findings.length : 0;

        animateCounter(metricTotalTests, total);
        animateCounter(metricAbnormalFlags, abnormal);
        animateCounter(metricCriticalCount, critCount);
        animateCounter(metricTokensUsed, data.tokens_used || 0);

        // Populate Table
        if (flagsTableBody) {
            flagsTableBody.innerHTML = '';
            (data.flags || []).forEach(row => {
                const tr = document.createElement('tr');
                const ref = (row.reference_low !== null && row.reference_high !== null)
                    ? `${row.reference_low} - ${row.reference_high} ${row.unit || ''}`
                    : 'Standard Baseline';

                tr.innerHTML = `
                    <td><strong>${escapeHtml(row.test)}</strong></td>
                    <td><code>${row.value} ${escapeHtml(row.unit || '')}</code></td>
                    <td>${escapeHtml(ref)}</td>
                    <td><span class="flag-badge flag-${row.flag}">${row.flag}</span></td>
                `;
                flagsTableBody.appendChild(tr);
            });
        }

        // Plain-Language Markdown Explanation
        if (explanationText) {
            explanationText.innerHTML = renderMarkdown(data.explanation || '');
        }

        // Update 3D Stage Info
        const overlayStat = document.getElementById('overlayAnalyteCount');
        if (overlayStat) {
            overlayStat.textContent = `Biomarkers: ${total} Evaluated (${abnormal} Flagged)`;
        }

        // Scroll to results
        resultsSection?.scrollIntoView({ behavior: 'smooth' });
    }

    function animateCounter(el, target) {
        if (!el) return;
        let current = 0;
        const step = Math.max(1, Math.floor(target / 15));
        const timer = setInterval(() => {
            current += step;
            if (current >= target) {
                current = target;
                clearInterval(timer);
            }
            el.textContent = current.toLocaleString();
        }, 30);
    }

    // ─── CHATBOT CONVERSATIONAL LOGIC ───
    sendBtn?.addEventListener('click', () => {
        const text = promptInput?.value.trim();
        if (text) {
            promptInput.value = '';
            handleUserChatMessage(text);
        }
    });

    promptInput?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            const text = promptInput?.value.trim();
            if (text) {
                promptInput.value = '';
                handleUserChatMessage(text);
            }
        }
    });

    chatBarSendBtn?.addEventListener('click', () => {
        const text = chatBarInput?.value.trim();
        if (text) {
            chatBarInput.value = '';
            handleUserChatMessage(text);
        }
    });

    chatBarInput?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            const text = chatBarInput?.value.trim();
            if (text) {
                chatBarInput.value = '';
                handleUserChatMessage(text);
            }
        }
    });

    chatClearBtn?.addEventListener('click', () => {
        if (chatMessagesContainer) {
            chatMessagesContainer.innerHTML = `
                <div class="chat-message message-assistant">
                    <div class="message-avatar">✚</div>
                    <div class="message-bubble">
                        <div class="message-content">
                            Chat cleared. How can I assist you with your lab report?
                        </div>
                    </div>
                </div>
            `;
        }
        state.chatHistory = [];
    });

    // Suggestion pills
    document.querySelectorAll('.pill-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const prompt = chip.getAttribute('data-prompt');
            if (prompt) {
                handleUserChatMessage(prompt);
            }
        });
    });

    async function handleUserChatMessage(userText) {
        if (state.isChatting) return;

        // Add user bubble
        appendChatMessage('user', userText);
        state.chatHistory.push({ role: 'user', content: userText });

        // Scroll into chat section
        document.getElementById('chatSection')?.scrollIntoView({ behavior: 'smooth' });

        // Show typing indicator
        if (chatTypingIndicator) chatTypingIndicator.style.display = 'flex';
        state.isChatting = true;

        try {
            const resp = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: userText,
                    report_text: state.activeReportText,
                    flags: state.activeFlags,
                    critical_findings: state.activeCritical,
                    history: state.chatHistory.slice(-6),
                    language: languageSelect?.value || 'English'
                })
            });

            if (!resp.ok) {
                throw new Error(`Chat request error: ${resp.status}`);
            }

            const data = await resp.json();
            appendChatMessage('assistant', data.reply);
            state.chatHistory.push({ role: 'assistant', content: data.reply });

        } catch (err) {
            appendChatMessage('assistant', `I had trouble connecting: ${err.message}. Please try again shortly.`);
        } finally {
            if (chatTypingIndicator) chatTypingIndicator.style.display = 'none';
            state.isChatting = false;
        }
    }

    function appendChatMessage(role, text) {
        if (!chatMessagesContainer) return;
        const msgDiv = document.createElement('div');
        msgDiv.className = `chat-message message-${role}`;

        const avatar = role === 'user' ? '👤' : '✚';
        msgDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-bubble">
                <div class="message-content">${renderMarkdown(text)}</div>
            </div>
        `;
        chatMessagesContainer.appendChild(msgDiv);
        chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
    }

    // ─── SPEECH TO TEXT DICTATION ───
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
        state.speechRecognition = new SpeechRec();
        state.speechRecognition.continuous = false;
        state.speechRecognition.interimResults = false;

        state.speechRecognition.onresult = (e) => {
            const transcript = e.results[0][0].transcript;
            if (promptInput) promptInput.value = transcript;
            if (chatBarInput) chatBarInput.value = transcript;
        };

        state.speechRecognition.onend = () => {
            state.isRecording = false;
            micBtn?.classList.remove('recording');
        };

        micBtn?.addEventListener('click', () => {
            if (state.isRecording) {
                state.speechRecognition.stop();
            } else {
                state.speechRecognition.start();
                state.isRecording = true;
                micBtn.classList.add('recording');
            }
        });
    } else {
        if (micBtn) micBtn.style.display = 'none';
    }

    // ─── THREE.JS 3D BIOMARKER VISUALIZATION ───
    function init3DBiomarkerVisualizer() {
        const canvas = document.getElementById('biomarker3DCanvas');
        if (!canvas || typeof THREE === 'undefined') return;

        const container = canvas.parentElement;
        const width = container.clientWidth || 320;
        const height = container.clientHeight || 280;

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        camera.position.z = 32;

        const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);

        const pointLight1 = new THREE.PointLight(0x38bdf8, 2, 60);
        pointLight1.position.set(10, 10, 10);
        scene.add(pointLight1);

        const pointLight2 = new THREE.PointLight(0xa855f7, 2, 60);
        pointLight2.position.set(-10, -10, -10);
        scene.add(pointLight2);

        // DNA Double Helix Generation
        const helixGroup = new THREE.Group();
        const sphereGeo = new THREE.SphereGeometry(0.5, 16, 16);
        const matBlue = new THREE.MeshPhongMaterial({ color: 0x38bdf8, shininess: 80 });
        const matPurple = new THREE.MeshPhongMaterial({ color: 0xc084fc, shininess: 80 });
        const matBond = new THREE.LineBasicMaterial({ color: 0x64748b, transparent: true, opacity: 0.5 });

        const count = 30;
        const radius = 5;
        const pitch = 0.55;

        for (let i = 0; i < count; i++) {
            const t = (i - count / 2) * pitch;
            const angle = i * 0.45;

            const x1 = Math.cos(angle) * radius;
            const z1 = Math.sin(angle) * radius;
            const x2 = Math.cos(angle + Math.PI) * radius;
            const z2 = Math.sin(angle + Math.PI) * radius;

            const sphere1 = new THREE.Mesh(sphereGeo, matBlue);
            sphere1.position.set(x1, t, z1);
            helixGroup.add(sphere1);

            const sphere2 = new THREE.Mesh(sphereGeo, matPurple);
            sphere2.position.set(x2, t, z2);
            helixGroup.add(sphere2);

            // Base pair connector
            const bondGeo = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(x1, t, z1),
                new THREE.Vector3(x2, t, z2)
            ]);
            const bond = new THREE.Line(bondGeo, matBond);
            helixGroup.add(bond);
        }

        scene.add(helixGroup);

        // Ambient Floating Particles
        const partCount = 70;
        const partGeo = new THREE.BufferGeometry();
        const positions = new Float32Array(partCount * 3);
        for (let i = 0; i < partCount * 3; i++) {
            positions[i] = (Math.random() - 0.5) * 35;
        }
        partGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const partMat = new THREE.PointsMaterial({ color: 0x93c5fd, size: 0.35, transparent: true, opacity: 0.7 });
        const particles = new THREE.Points(partGeo, partMat);
        scene.add(particles);

        // Mouse tilt effect
        let mouseX = 0;
        let mouseY = 0;
        container.addEventListener('mousemove', (e) => {
            const rect = container.getBoundingClientRect();
            mouseX = (e.clientX - rect.left - width / 2) * 0.001;
            mouseY = (e.clientY - rect.top - height / 2) * 0.001;
        });

        // Animation Loop
        function animate() {
            requestAnimationFrame(animate);
            helixGroup.rotation.y += 0.015;
            helixGroup.rotation.x += (mouseY - helixGroup.rotation.x) * 0.05;
            helixGroup.rotation.z += (mouseX - helixGroup.rotation.z) * 0.05;

            particles.rotation.y -= 0.003;
            renderer.render(scene, camera);
        }
        animate();

        // Resize Listener
        window.addEventListener('resize', () => {
            const w = container.clientWidth;
            const h = container.clientHeight;
            if (w && h) {
                camera.aspect = w / h;
                camera.updateProjectionMatrix();
                renderer.setSize(w, h);
            }
        });
    }

    // ─── SIMPLE MARKDOWN RENDERER ───
    function renderMarkdown(text) {
        if (!text) return '';
        let html = escapeHtml(text);

        // Headers
        html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
        html = html.replace(/^## (.*$)/gim, '<h3>$1</h3>');

        // Bold
        html = html.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');

        // Code
        html = html.replace(/`(.*?)`/gim, '<code>$1</code>');

        // Blockquotes
        html = html.replace(/^\> (.*$)/gim, '<blockquote>$1</blockquote>');

        // Bullets
        html = html.replace(/^\• (.*$)/gim, '<li>$1</li>');
        html = html.replace(/^- (.*$)/gim, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>)/gim, '<ul>$1</ul>');

        // Line breaks
        html = html.replace(/\n\n/g, '<br><br>');

        return html;
    }

    function escapeHtml(str) {
        if (typeof str !== 'string') return String(str);
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }
});
