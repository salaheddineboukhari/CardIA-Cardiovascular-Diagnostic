// ==================== UTILITAIRES ====================
function showLoading() {
    document.body.classList.add('loading');
}

function hideLoading() {
    document.body.classList.remove('loading');
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// ==================== GESTION DES TABS ====================
document.addEventListener('DOMContentLoaded', function() {
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            // Enlever la classe active de tous les tabs
            tabs.forEach(t => t.classList.remove('tab-active'));
            // Ajouter la classe active au tab cliqué
            this.classList.add('tab-active');
            
            const tabName = this.dataset.tab;
            if (tabName === 'simulation') {
                window.location.href = '/simulate';
            } else if (tabName === 'api') {
                window.location.href = '/api/predict';
            }
        });
    });
});

// ==================== GRAPHIQUES ====================
function initRiskGauge(riskValue) {
    const ctx = document.getElementById('riskGauge');
    if (!ctx) return;
    
    const riskPercent = riskValue * 100;
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['RISK', 'SAFE'],
            datasets: [{
                data: [riskPercent, 100 - riskPercent],
                backgroundColor: [
                    riskPercent < 30 ? '#059669' : (riskPercent < 60 ? '#d97706' : '#e11d48'),
                    '#e9ecef'
                ],
                borderWidth: 0
            }]
        },
        options: {
            cutout: '70%',
            plugins: {
                tooltip: { enabled: false },
                legend: { display: false }
            }
        }
    });
}

function initRiskDistribution(data) {
    const ctx = document.getElementById('riskDistributionChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Low Risk', 'Moderate Risk', 'High Risk'],
            datasets: [{
                data: [data.low || 0, data.moderate || 0, data.high || 0],
                backgroundColor: ['#059669', '#d97706', '#e11d48'],
                borderWidth: 0
            }]
        },
        options: {
            plugins: {
                legend: { position: 'bottom' }
            }
        }
    });
}

function initTrendsChart(data) {
    const ctx = document.getElementById('trendsChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.labels || ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            datasets: [{
                label: 'Average Risk',
                data: data.values || [0.3, 0.35, 0.32, 0.4, 0.38, 0.42],
                borderColor: '#0d9488',
                backgroundColor: 'rgba(13,148,136,0.08)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        callback: value => (value * 100) + '%'
                    }
                }
            }
        }
    });
}

function initEvolutionChart(dates, predictions) {
    const ctx = document.getElementById('evolutionChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'Predicted Risk',
                data: predictions,
                borderColor: '#0d9488',
                backgroundColor: 'rgba(13,148,136,0.08)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        callback: value => (value * 100) + '%'
                    }
                }
            }
        }
    });
}

// ==================== EXPLICATION ====================
function showExplanation() {
    const section = document.getElementById('explanationSection');
    const content = document.getElementById('explanationContent');
    
    if (section && content) {
        if (section.classList.contains('hidden')) {
            showLoading();
            
            fetch('/explain-prediction', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            })
            .then(response => response.json())
            .then(data => {
                content.innerHTML = '';
                if (data.explanations && data.explanations.length > 0) {
                    data.explanations.forEach(exp => {
                        const p = document.createElement('p');
                        p.className = 'explanation-item';
                        p.innerHTML = exp;
                        content.appendChild(p);
                    });
                } else {
                    content.innerHTML = '<p class="explanation-item">No explanation available</p>';
                }
                
                section.classList.remove('hidden');
                hideLoading();
            })
            .catch(error => {
                console.error('Error:', error);
                hideLoading();
                showNotification('Error loading explanation', 'error');
            });
        } else {
            section.classList.add('hidden');
        }
    }
}

// ==================== RECOMMANDATIONS ====================
function showRecommendations() {
    const section = document.getElementById('recommendationsSection');
    const content = document.getElementById('recommendationsContent');
    
    if (section && content) {
        if (section.classList.contains('hidden')) {
            showLoading();
            
            fetch('/get-recommendations', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            })
            .then(response => response.json())
            .then(recommendations => {
                content.innerHTML = '';
                
                if (recommendations && recommendations.length > 0) {
                    recommendations.forEach(rec => {
                        const card = document.createElement('div');
                        card.className = 'recommendation-card';
                        
                        card.innerHTML = `
                            <h4 class="rec-category">${rec.category || 'Recommendation'}</h4>
                            <p class="rec-advice">${rec.advice || ''}</p>
                            <ul class="rec-actions">
                                ${rec.actions ? rec.actions.map(action => `<li>${action}</li>`).join('') : ''}
                            </ul>
                            <div class="rec-meta">
                                <span class="rec-improvement">📉 ${rec.expected_improvement || 'N/A'}</span>
                                <span class="rec-timeframe">⏱️ ${rec.timeframe || 'N/A'}</span>
                            </div>
                        `;
                        
                        content.appendChild(card);
                    });
                } else {
                    content.innerHTML = '<p class="no-data">No recommendations available</p>';
                }
                
                section.classList.remove('hidden');
                hideLoading();
            })
            .catch(error => {
                console.error('Error:', error);
                hideLoading();
                showNotification('Error loading recommendations', 'error');
            });
        } else {
            section.classList.add('hidden');
        }
    }
}

// ==================== ÉVOLUTION ====================
function showEvolution() {
    const section = document.getElementById('evolutionSection');
    const message = document.getElementById('evolutionMessage');
    
    if (section && message) {
        if (section.classList.contains('hidden')) {
            showLoading();
            
            fetch('/predict-evolution', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    message.textContent = data.error;
                } else {
                    if (data.dates && data.predictions) {
                        initEvolutionChart(data.dates, data.predictions);
                    }
                    message.textContent = data.message || 'Evolution predicted';
                }
                
                section.classList.remove('hidden');
                hideLoading();
            })
            .catch(error => {
                console.error('Error:', error);
                hideLoading();
                showNotification('Error loading evolution', 'error');
            });
        } else {
            section.classList.add('hidden');
        }
    }
}

// ==================== RAPPORT PDF ====================
function downloadReport() {
    showLoading();
    
    fetch('/generate-report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(response => response.blob())
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'cardiovascular_report.pdf';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        a.remove();
        
        hideLoading();
        showNotification('Report downloaded successfully', 'success');
    })
    .catch(error => {
        console.error('Error:', error);
        hideLoading();
        showNotification('Error downloading report', 'error');
    });
}

// ==================== VALIDATION FORMULAIRE ====================
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('patientForm');
    if (form) {
        form.addEventListener('submit', function(e) {
            // Récupérer les valeurs et les convertir en nombres
            const age = parseInt(document.querySelector('input[name="age"]').value);
            const weight = parseFloat(document.querySelector('input[name="weight"]').value);
            const height = parseFloat(document.querySelector('input[name="height"]').value);
            const ap_hi = parseFloat(document.querySelector('input[name="ap_hi"]').value);
            const ap_lo = parseFloat(document.querySelector('input[name="ap_lo"]').value);
            
            // Validation de base
            if (weight < 30 || weight > 200) {
                e.preventDefault();
                showNotification('Weight must be between 30 and 200 kg', 'error');
                return false;
            }
            
            if (height < 100 || height > 250) {
                e.preventDefault();
                showNotification('Height must be between 100 and 250 cm', 'error');
                return false;
            }
            
            // CORRECTION: Comparaison numérique au lieu de string
            if (ap_hi <= ap_lo) {
                e.preventDefault();
                showNotification('Systolic pressure must be greater than diastolic', 'error');
                console.log('Validation failed:', ap_hi, '<=', ap_lo); // Pour debug
                return false;
            }
            
            console.log('Validation passed:', ap_hi, '>', ap_lo); // Pour debug
            showLoading();
        });
    }
});

// ==================== DASHBOARD ACTIONS ====================
function viewPatient(id) {
    if (id) {
        window.location.href = `/patient-history/${id}`;
    }
}

function exportData() {
    showNotification('Exporting data...', 'info');
    // Implémenter l'export
    setTimeout(() => {
        showNotification('Data exported successfully', 'success');
    }, 1500);
}

function generateReport() {
    showNotification('Generating report...', 'info');
    // Implémenter la génération de rapport
    setTimeout(() => {
        showNotification('Report generated successfully', 'success');
    }, 1500);
}

function refreshDashboard() {
    location.reload();
}

// ==================== ANIMATIONS ====================
document.addEventListener('DOMContentLoaded', function() {
    // Animation au scroll
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    });
    
    document.querySelectorAll('.metric-card, .stat-card, .recommendation-card').forEach(el => {
        observer.observe(el);
    });
});

// ==================== GESTION DES ERREURS ====================
window.onerror = function(msg, url, lineNo, columnNo, error) {
    console.error('Error: ' + msg + '\nURL: ' + url + '\nLine: ' + lineNo);
    showNotification('An error occurred. Please try again.', 'error');
    return false;
};

// ==================== INITIALISATION ====================
document.addEventListener('DOMContentLoaded', function() {
    console.log('JavaScript loaded successfully');
    
    // Vérifier si on est sur la page des résultats
    if (document.getElementById('riskGauge')) {
        console.log('Risk gauge found');
    }
});
// ==================== RAPPORT DE SIMULATION ====================
function downloadSimulationReport() {
    // Vérifier que le bouton existe
    const downloadBtn = event.target;
    const originalText = downloadBtn.innerHTML;
    downloadBtn.innerHTML = '⏳ Génération...';
    downloadBtn.disabled = true;

    // Récupérer les données depuis les attributs data du bouton ou depuis des variables globales
    const reportData = {
        base_patient: window.basePatientData || {},
        modifications: window.modificationsData || {},
        result: window.resultData || {},
        base_result: window.baseResultData || {},
        improvement: window.improvementData || 0
    };

    // Appel à l'API pour générer le rapport
    fetch('/generate-simulation-report', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(reportData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Erreur réseau');
        }
        return response.blob();
    })
    .then(blob => {
        // Créer un lien de téléchargement
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `simulation_rapport_${new Date().toISOString().slice(0,10)}.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        a.remove();
        
        // Afficher le message de succès
        showNotification('Rapport téléchargé avec succès', 'success');
    })
    .catch(error => {
        console.error('Erreur:', error);
        showNotification('Erreur lors du téléchargement du rapport', 'error');
    })
    .finally(() => {
        // Restaurer le bouton
        downloadBtn.innerHTML = originalText;
        downloadBtn.disabled = false;
    });
}

// ==================== GRAPHIQUE DE COMPARAISON ====================
function initComparisonChart(baseRisk, simulatedRisk) {
    const ctx = document.getElementById('comparisonChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Risque initial', 'Risque simulé'],
            datasets: [{
                label: 'Niveau de risque (%)',
                data: [baseRisk * 100, simulatedRisk * 100],
                backgroundColor: [
                    'rgba(225,29,72,0.5)',
                    'rgba(13,148,136,0.5)'
                ],
                borderColor: [
                    'rgba(225,29,72,1)',
                    'rgba(13,148,136,1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}

// ==================== XRAY ANALYSIS ====================
let xrayImageBase64 = null;

function initXrayDropzone() {
    const dropZone = document.getElementById('dropZone');
    if (!dropZone) return;

    dropZone.addEventListener('dragover', e => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', e => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files[0]) loadXrayImage(e.dataTransfer.files[0]);
    });

    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.addEventListener('change', e => {
            if (e.target.files[0]) loadXrayImage(e.target.files[0]);
        });
    }
}

function loadXrayImage(file) {
    const reader = new FileReader();
    reader.onload = e => {
        xrayImageBase64 = e.target.result.split(',')[1];
        const img = document.getElementById('previewImg');
        if (img) { img.src = e.target.result; img.style.display = 'block'; }
        const btn = document.getElementById('btnAnalyze');
        if (btn) btn.style.display = 'block';
        const card = document.getElementById('resultCard');
        if (card) card.style.display = 'none';
        const ph = document.getElementById('placeholder');
        if (ph) ph.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

async function analyzeXray() {
    const patientEl = document.getElementById('patientSelect');
    if (!patientEl) return;
    const patientName = patientEl.value;
    if (!patientName) { alert('Veuillez sélectionner un patient'); return; }
    if (!xrayImageBase64) { alert('Veuillez sélectionner une image'); return; }

    const loading = document.getElementById('loading');
    const btn = document.getElementById('btnAnalyze');
    if (loading) loading.style.display = 'block';
    if (btn) btn.style.display = 'none';

    try {
        const response = await fetch('/analyze-xray', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: xrayImageBase64, patient_name: patientName })
        });
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        showXrayResult(data);
    } catch(e) {
        alert('Erreur : ' + e.message);
        const ph = document.getElementById('placeholder');
        if (ph) ph.style.display = 'flex';
    } finally {
        if (loading) loading.style.display = 'none';
        if (btn) btn.style.display = 'block';
    }
}

function showXrayResult(data) {
    const card = document.getElementById('resultCard');
    const badge = document.getElementById('statusBadge');
    const anomalieList = document.getElementById('anomalieList');
    const scoresBars = document.getElementById('scoresBars');
    if (!card) return;

    const isNormal = data.status === 'normal';
    if (badge) {
        badge.textContent = isNormal ? ' Normal' : ' Anomalie détectée';
        badge.className = 'xray-status-badge ' + (isNormal ? 'xray-status-normal' : 'xray-status-anomalie');
    }

    if (anomalieList) {
        anomalieList.innerHTML = '';
        if (isNormal) {
            anomalieList.innerHTML = '<p class="xray-normal-msg"> Aucune anomalie cardiaque significative détectée.</p>';
        } else if (data.anomalies && Object.keys(data.anomalies).length > 0) {
            Object.entries(data.anomalies).forEach(([k, v]) => {
                anomalieList.innerHTML += `<div class="xray-anomalie-item">⚠️ <span><strong>${k}</strong> - score : ${v}%</span></div>`;
            });
        }
    }

    if (scoresBars && data.scores) {
        const colors = {
            Cardiomegaly: '#fc8181', Edema: '#f6ad55',
            Consolidation: '#f6ad55', Pneumonia: '#fc8181',
            'No Finding': '#68d391'
        };
        scoresBars.innerHTML = '';
        Object.entries(data.scores).forEach(([label, score]) => {
            scoresBars.innerHTML += `
                <div class="xray-score-bar">
                    <div class="xray-score-label"><span>${label}</span><span>${score}%</span></div>
                    <div class="xray-score-track">
                        <div class="xray-score-fill" style="width:${score}%;background:${colors[label]||'#90cdf4'};"></div>
                    </div>
                </div>`;
        });
    }

    card.style.display = 'block';
}

// ==================== AGE CALCULATION ====================
function calculateAge(birthDate) {
    if (!birthDate) return '';
    const today = new Date();
    const birth = new Date(birthDate);
    let age = today.getFullYear() - birth.getFullYear();
    if ((today.getMonth() - birth.getMonth()) < 0 ||
        (today.getMonth() === birth.getMonth() && today.getDate() < birth.getDate())) age--;
    return age;
}

// ==================== QUICK ADD PATIENT MODAL ====================
function initQuickAddModal() {
    const quickModal   = document.getElementById('quickModal');
    const modalOverlay = document.getElementById('modalOverlay');
    const quickAddBtn  = document.getElementById('quickAddBtn');
    if (!quickModal || !modalOverlay) return;

    if (quickAddBtn) {
        quickAddBtn.addEventListener('click', function() {
            quickModal.classList.add('open');
            modalOverlay.classList.add('open');
            document.getElementById('newPatientName').focus();
        });
    }

    modalOverlay.addEventListener('click', closeQuickModal);
}

function closeQuickModal() {
    const quickModal   = document.getElementById('quickModal');
    const modalOverlay = document.getElementById('modalOverlay');
    if (quickModal) quickModal.classList.remove('open');
    if (modalOverlay) modalOverlay.classList.remove('open');
    const name = document.getElementById('newPatientName');
    const birth = document.getElementById('newPatientBirth');
    if (name) name.value = '';
    if (birth) birth.value = '';
}

function saveQuickPatient() {
    const name  = document.getElementById('newPatientName').value.trim();
    const birth = document.getElementById('newPatientBirth').value;
    if (!name) { alert('Le nom du patient est requis.'); return; }

    fetch('/quick-add-patient', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ full_name: name, birth_date: birth })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) { location.reload(); }
        else { alert('Erreur : ' + (data.error || 'Impossible de créer le patient')); }
    })
    .catch(() => alert('Erreur réseau lors de la création du patient.'));
}

// ==================== PATIENT SELECT AGE AUTO ====================
function initPatientSelectAge() {
    const patientSelect = document.getElementById('patientSelect');
    if (!patientSelect || patientSelect.tagName !== 'SELECT') return;

    patientSelect.addEventListener('change', function() {
        const opt  = this.options[this.selectedIndex];
        const birth = opt.getAttribute('data-birth');
        const age   = calculateAge(birth);
        const ageField = document.getElementById('age');
        const birthField = document.getElementById('birth_date');
        if (ageField) ageField.value = age || '';
        if (birthField) birthField.value = birth || '';
    });

    // Init on load
    if (patientSelect.value) {
        const opt = patientSelect.options[patientSelect.selectedIndex];
        const birth = opt.getAttribute('data-birth');
        const age = calculateAge(birth);
        const ageField = document.getElementById('age');
        const birthField = document.getElementById('birth_date');
        if (age && ageField) ageField.value = age;
        if (birth && birthField) birthField.value = birth;
    }
}

// ==================== CHANGE PASSWORD VALIDATION ====================
function initChangePasswordForm() {
    const form = document.getElementById('changePasswordForm');
    if (!form) return;
    form.addEventListener('submit', function(e) {
        const np = document.getElementById('newPwd').value;
        const cp = document.getElementById('confirmPwd').value;
        const err = document.getElementById('pwdError');
        if (np !== cp) {
            e.preventDefault();
            if (err) err.style.display = 'block';
        } else {
            if (err) err.style.display = 'none';
        }
    });
}

// ==================== REGISTER BIRTH DATE TOGGLE ====================
function initRegisterForm() {
    const roleSelect = document.getElementById('roleSelect');
    const birthDateGroup = document.getElementById('birthDateGroup');
    const birthDateInput = document.getElementById('birthDateInput');
    if (!roleSelect) return;

    if (birthDateInput) {
        birthDateInput.setAttribute('max', new Date().toISOString().split('T')[0]);
    }

    function toggleBirthDate() {
        if (roleSelect.value === 'patient') {
            if (birthDateGroup) birthDateGroup.classList.add('visible');
            if (birthDateInput) birthDateInput.setAttribute('required', 'required');
        } else {
            if (birthDateGroup) birthDateGroup.classList.remove('visible');
            if (birthDateInput) { birthDateInput.removeAttribute('required'); birthDateInput.value = ''; }
        }
    }

    roleSelect.addEventListener('change', toggleBirthDate);
    toggleBirthDate();
}

// ==================== DASHBOARD FILTER ====================
function filterTable() {
    const selectedPatient = document.getElementById('patientFilter') ? document.getElementById('patientFilter').value : 'all';
    const selectedType    = document.getElementById('typeFilter') ? document.getElementById('typeFilter').value : 'all';
    const selectedRisk    = document.getElementById('riskFilter') ? document.getElementById('riskFilter').value : 'all';

    document.querySelectorAll('.patient-group').forEach(group => {
        let showGroup = false;
        group.querySelectorAll('tbody tr').forEach(row => {
            let showRow = true;
            const patientName = group.dataset.patient;
            if (selectedPatient !== 'all' && patientName !== selectedPatient) showRow = false;
            const rowType = row.dataset.type;
            if (selectedType !== 'all' && rowType !== selectedType) showRow = false;
            if (selectedRisk !== 'all' && rowType === 'risk') {
                const risk = parseFloat(row.dataset.risk);
                if (selectedRisk === 'high' && risk <= 0.6) showRow = false;
                if (selectedRisk === 'moderate' && (risk <= 0.3 || risk > 0.6)) showRow = false;
                if (selectedRisk === 'low' && risk > 0.3) showRow = false;
            }
            row.style.display = showRow ? '' : 'none';
            if (showRow) showGroup = true;
        });
        group.style.display = showGroup ? '' : 'none';
    });
}

// ==================== AUTO INIT ON DOM READY ====================
document.addEventListener('DOMContentLoaded', function() {
    initXrayDropzone();
    initQuickAddModal();
    initPatientSelectAge();
    initChangePasswordForm();
    initRegisterForm();
});