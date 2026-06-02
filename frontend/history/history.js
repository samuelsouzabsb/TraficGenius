document.addEventListener('DOMContentLoaded', () => {
    // Inicializa Relógio
    startClock();
    
    // Inicializa Mapa
    initMap();

    // Setup do Picker de Cores Global
    const colorPicker = document.getElementById('theme-color-picker');
    if (colorPicker) {
        colorPicker.addEventListener('input', (e) => {
            document.documentElement.style.setProperty('--neon-cyan', e.target.value);
            // Atualiza cores dos gráficos
            Chart.helpers.each(Chart.instances, function(instance){
                instance.update();
            });
        });
    }
    
    // Buscar Dados
    fetch('historical_data.json')
        .then(res => res.json())
        .then(data => {
            // Render premium charts
            try { renderWeatherImpact(data); } catch(e) { console.error('Error rendering weather impact:', e); }
            try { renderRoadFeatures(data); } catch(e) { console.error('Error rendering road features:', e); }
            try { renderTimeHeatmap(data); } catch(e) { console.error('Error rendering time heatmap:', e); }
            
            // Render original charts
            try { if(data.time_vs_sun) renderTimeChart(data.time_vs_sun); } catch(e) { console.error('Error rendering time chart:', e); }
            try { if(data.weather_matrix) renderWeatherChart(data.weather_matrix); } catch(e) { console.error('Error rendering weather chart:', e); }
            try { if(data.infra_radar) renderInfraRadar(data.infra_radar); } catch(e) { console.error('Error rendering infra radar:', e); }
            try { if(data.severity_donut) renderSeverityDonut(data.severity_donut); } catch(e) { console.error('Error rendering donut:', e); }
            try { if(data.distance_hist) renderDistanceBar(data.distance_hist); } catch(e) { console.error('Error rendering distance bar:', e); }
        })
        .catch(err => {
            console.error("Error loading history data, loading mock fallback:", err);
            loadMockPremiumData();
        });
});

function loadMockPremiumData() {
    // New Premium Charts
    try { renderWeatherImpact(); } catch(e) { console.error(e); }
    try { renderRoadFeatures(); } catch(e) { console.error(e); }
    try { renderTimeHeatmap(); } catch(e) { console.error(e); }
    
    // Old Original Charts
    loadMockHistoryData();
}

function loadMockHistoryData() {
    const mockData = {
        time_vs_sun: {
            labels: Array.from({length: 24}, (_, i) => i + "h"),
            day: Array.from({length: 24}, () => Math.floor(Math.random() * 100 + 50)),
            night: Array.from({length: 24}, () => Math.floor(Math.random() * 50 + 20))
        },
        weather_matrix: Array.from({length: 50}, () => ({
            temp: Math.random() * 80 + 20,
            vis: Math.random() * 10,
            hum: Math.random() * 100,
            sev: Math.floor(Math.random() * 4) + 1
        })),
        infra_radar: {
            labels: ["Crossing", "Junction", "Traffic_Signal", "Station", "Stop"],
            values: [1500, 2300, 3100, 400, 800]
        },
        severity_donut: {
            labels: ["G1", "G2", "G3", "G4"],
            values: [5000, 15000, 3000, 800]
        },
        distance_hist: {
            labels: ["< 1mi", "1-3mi", "3-5mi", "> 5mi"],
            values: [12000, 5000, 1500, 500]
        }
    };
    renderTimeChart(mockData.time_vs_sun);
    renderWeatherChart(mockData.weather_matrix);
    renderInfraRadar(mockData.infra_radar);
    renderSeverityDonut(mockData.severity_donut);
    renderDistanceBar(mockData.distance_hist);
}

function startClock() {
    function update() {
        const now = new Date();
        const timeStr = now.toLocaleTimeString('en-GB');
        const clockEl = document.getElementById('live-clock');
        if (clockEl) clockEl.textContent = timeStr;
        
        const dateStr = now.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
        const parts = dateStr.split(' ');
        const formattedDate = `${parts[0]}<br>${parts[1]} ${parts[2]}`;
        const dateEl = document.getElementById('live-date');
        if (dateEl) dateEl.innerHTML = formattedDate;
    }
    update();
    setInterval(update, 1000);
}

function initMap() {
    const mapEl = document.getElementById('history-map');
    if (!mapEl) return;
    
    const map = L.map('history-map').setView([39.8283, -98.5795], 4);
    
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://carto.com/attributions">CARTO</a>'
    }).addTo(map);

    const mockClusters = [
        { lat: 34.05, lng: -118.24, severity: 4 }, // LA
        { lat: 40.71, lng: -74.00, severity: 2 }, // NY
        { lat: 41.87, lng: -87.62, severity: 3 }, // Chicago
        { lat: 29.76, lng: -95.36, severity: 4 }, // Houston
        { lat: 33.74, lng: -84.38, severity: 1 }  // Atlanta
    ];

    const interactiveList = document.getElementById('interactive-incident-list');
    const cities = ["Los Angeles, CA", "New York, NY", "Chicago, IL", "Houston, TX", "Atlanta, GA"];
    const types = ["Colisão Múltipla", "Pista Escorregadia", "Falha de Semáforo", "Obstrução Grave", "Acidente Leve"];

    mockClusters.forEach((c, index) => {
        let color = '#34d399'; // G1
        if (c.severity === 2) color = '#fbbf24'; // G2
        if (c.severity === 3) color = '#f97316'; // G3
        if (c.severity === 4) color = '#ef4444'; // G4

        const marker = L.circleMarker([c.lat, c.lng], {
            radius: c.severity * 5,
            fillColor: color,
            color: color,
            weight: 1,
            opacity: 0.8,
            fillOpacity: 0.5
        }).addTo(map);

        c.city = cities[index];
        c.type = types[index];
        
        const popupContent = `
            <div style="font-family: 'Inter', sans-serif; color: #333;">
                <h4 style="margin: 0 0 5px 0; color: #1a1a2e; font-size: 14px;">${c.type}</h4>
                <p style="margin: 0; font-size: 12px; color: #666;"><b>Local:</b> ${c.city}</p>
                <p style="margin: 0; font-size: 12px; color: #666;"><b>Severidade:</b> G${c.severity}</p>
            </div>
        `;
        marker.bindPopup(popupContent);

        // Add to interactive list
        if (interactiveList) {
            const item = document.createElement('div');
            item.className = 'incident-item';
            item.innerHTML = `
                <div class="inc-title">
                    <span>${c.type}</span>
                    <span class="sev-badge g${c.severity}">G${c.severity}</span>
                </div>
                <div class="inc-desc">${c.city}</div>
            `;
            item.addEventListener('click', () => {
                // Remove selected class from all
                document.querySelectorAll('.incident-item').forEach(el => el.style.borderColor = 'rgba(255, 255, 255, 0.1)');
                // Highlight this one
                item.style.borderColor = color;
                
                // Fly to marker and open popup
                map.flyTo([c.lat, c.lng], 13, { animate: true, duration: 1.5 });
                setTimeout(() => marker.openPopup(), 1500);
            });
            interactiveList.appendChild(item);
        }
    });
}

Chart.defaults.color = '#8B9BB4';
Chart.defaults.font.family = "'Inter', sans-serif";

function getCyan() {
    return getComputedStyle(document.documentElement).getPropertyValue('--neon-cyan').trim() || '#00e5ff';
}

function renderWeatherImpact() {
    const ctx = document.getElementById('weatherImpactChart').getContext('2d');
    
    // Generate mock data for the 3 clusters
    const scatterData = [];
    
    // Cyan cluster (Low Temp, Low/Med Precip)
    for(let i=0; i<15; i++) {
        scatterData.push({
            x: Math.random() * 20 - 10, 
            y: Math.random() * 10 + 5, 
            r: Math.random() * 15 + 5,
            cluster: 'cyan'
        });
    }
    // Orange cluster (Med Temp, Med Precip)
    for(let i=0; i<12; i++) {
        scatterData.push({
            x: Math.random() * 15 + 10, 
            y: Math.random() * 12 + 6, 
            r: Math.random() * 12 + 5,
            cluster: 'orange'
        });
    }
    // Red cluster (High Temp, High Precip)
    for(let i=0; i<6; i++) {
        scatterData.push({
            x: Math.random() * 15 + 20, 
            y: Math.random() * 10 + 18, 
            r: Math.random() * 15 + 8,
            cluster: 'red'
        });
    }

    new Chart(ctx, {
        type: 'bubble',
        data: {
            datasets: [{
                label: 'Weather Impact',
                data: scatterData,
                backgroundColor: (context) => {
                    if (context.type !== 'data') return 'transparent';
                    const cluster = context.raw?.cluster;
                    if (cluster === 'cyan') return 'rgba(0, 229, 255, 0.3)';
                    if (cluster === 'orange') return 'rgba(255, 138, 0, 0.3)';
                    if (cluster === 'red') return 'rgba(239, 68, 68, 0.3)';
                    return '#00e5ff';
                },
                borderColor: (context) => {
                    if (context.type !== 'data') return 'transparent';
                    const cluster = context.raw?.cluster;
                    if (cluster === 'cyan') return 'rgba(0, 229, 255, 1)';
                    if (cluster === 'orange') return 'rgba(255, 138, 0, 1)';
                    if (cluster === 'red') return 'rgba(239, 68, 68, 1)';
                    return '#00e5ff';
                },
                borderWidth: 1.5
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    type: 'linear', position: 'bottom',
                    title: { display: true, text: 'Temperature (°C)', color: '#8B9BB4' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    min: -20, max: 40
                },
                y: {
                    type: 'linear', position: 'left',
                    title: { display: true, text: 'Precipitation (mm)', color: '#8B9BB4' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    min: 0, max: 30
                }
            }
        }
    });
}

function renderRoadFeatures() {
    const ctx = document.getElementById('roadFeaturesChart').getContext('2d');
    
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Junctions', 'Crossings', 'Traffic Signals', 'Speed Bumps', 'Roundabouts'],
            datasets: [{
                label: 'Influential Factors',
                data: [85, 85, 78, 78, 78],
                backgroundColor: 'rgba(0, 229, 255, 0.2)',
                borderColor: '#FF8A00',
                borderWidth: 2,
                pointBackgroundColor: '#00e5ff',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#00e5ff'
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                r: {
                    angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    pointLabels: { color: '#e2e8f0', font: { size: 12 } },
                    ticks: { display: false, max: 100, min: 0 }
                }
            }
        }
    });
}

function renderTimeHeatmap() {
    const ctx = document.getElementById('timeHeatmapChart').getContext('2d');
    
    const days = ['Sun', 'Sat', 'Fri', 'Thu', 'Wed', 'Tue', 'Mon']; // Y-axis inverted naturally
    const hours = Array.from({length: 24}, (_, i) => i < 10 ? '0'+i : ''+i);
    
    const heatmapData = [];
    for (let d = 0; d < 7; d++) {
        for (let h = 0; h < 24; h++) {
            // Create high risk clusters around 5-9 and 17-20
            let isRushHour = (h >= 5 && h <= 9) || (h >= 17 && h <= 20);
            let val = Math.random() * 20 + 10; // base value
            if (isRushHour && d > 1) { // Weekdays rush hour
                val += Math.random() * 50 + 40; 
            }
            heatmapData.push({
                x: hours[h],
                y: days[d],
                v: val
            });
        }
    }

    new Chart(ctx, {
        type: 'matrix',
        data: {
            datasets: [{
                label: 'Accident Density',
                data: heatmapData,
                backgroundColor(context) {
                    if (context.type !== 'data') {
                        return 'transparent';
                    }
                    const value = context.dataset.data[context.dataIndex].v;
                    const max = 120;
                    const min = 10;
                    const normalized = (value - min) / (max - min); // 0 to 1
                    
                    // Cyan (#00e5ff) to Orange (#FF8A00) gradient
                    // Cyan: RGB(0, 229, 255)
                    // Orange: RGB(255, 138, 0)
                    const r = Math.round(0 + (255 - 0) * normalized);
                    const g = Math.round(229 + (138 - 229) * normalized);
                    const b = Math.round(255 + (0 - 255) * normalized);
                    return `rgba(${r}, ${g}, ${b}, 0.8)`;
                },
                width: ({chart}) => (chart.chartArea || {}).width / 24 - 1,
                height: ({chart}) => (chart.chartArea || {}).height / 7 - 1,
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title() { return ''; },
                        label(context) {
                            const v = context.dataset.data[context.dataIndex];
                            return `Day: ${v.y}, Hour: ${v.x}, Incidents: ${Math.round(v.v)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'category',
                    labels: hours,
                    grid: { display: false },
                    title: { display: true, text: 'Hour', color: '#8B9BB4' },
                    ticks: { color: '#8B9BB4' }
                },
                y: {
                    type: 'category',
                    labels: days,
                    grid: { display: false },
                    title: { display: true, text: 'Day of Week', color: '#8B9BB4' },
                    ticks: { color: '#8B9BB4' }
                }
            }
        }
    });
}

function renderTimeChart(data) {
    const ctx = document.getElementById('timeLineChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.labels,
            datasets: [
                {
                    label: 'Dia',
                    data: data.day,
                    borderColor: '#FF8A00',
                    backgroundColor: 'rgba(255, 138, 0, 0.2)',
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'Noite',
                    data: data.night,
                    borderColor: '#3B82F6',
                    backgroundColor: 'rgba(59, 130, 246, 0.2)',
                    fill: true,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { position: 'top' } },
            scales: {
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, beginAtZero: true },
                x: { grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });
}

function renderWeatherChart(data) {
    const ctx = document.getElementById('weatherScatterChart').getContext('2d');
    
    // Bubble chart needs x, y, r
    const scatterData = data.map(d => ({ x: d.temp, y: d.vis, r: Math.max(d.hum / 8, 3) }));
    
    new Chart(ctx, {
        type: 'bubble',
        data: {
            datasets: [{
                label: 'Temp vs Visibilidade (Raio = Umidade)',
                data: scatterData,
                backgroundColor: 'rgba(0, 229, 255, 0.5)',
                borderColor: '#00e5ff'
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { title: { display: true, text: 'Visibilidade (mi)' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                x: { title: { display: true, text: 'Temperatura (F)' }, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });
}

function renderInfraRadar(data) {
    const ctx = document.getElementById('infraRadarChart').getContext('2d');
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: data.labels,
            datasets: [{
                label: 'Ocorrências Históricas',
                data: data.values,
                backgroundColor: 'rgba(255, 42, 85, 0.2)',
                borderColor: '#FF2A55',
                pointBackgroundColor: '#FF2A55'
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    pointLabels: { color: '#8B9BB4' },
                    ticks: { display: false }
                }
            }
        }
    });
}

function renderSeverityDonut(data) {
    const ctx = document.getElementById('severityDonutChart').getContext('2d');
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: data.labels,
            datasets: [{
                data: data.values,
                backgroundColor: ['#00e5ff', '#3B82F6', '#FF8A00', '#FF2A55'],
                borderWidth: 0,
                hoverOffset: 10
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { position: 'right' } },
            cutout: '70%'
        }
    });
}

function renderDistanceBar(data) {
    const ctx = document.getElementById('distanceBarChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.labels,
            datasets: [{
                label: 'Acidentes',
                data: data.values,
                backgroundColor: 'rgba(0, 229, 255, 0.5)',
                borderColor: '#00e5ff',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, beginAtZero: true },
                x: { grid: { display: false } }
            }
        }
    });
}
