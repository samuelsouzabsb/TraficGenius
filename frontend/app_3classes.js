/**
 * @file app_3classes.js
 * @description Script de inicialização e controle principal do Dashboard TraficGenius (3 Classes).
 */

document.addEventListener('DOMContentLoaded', () => {
    initMap();

    const colorPicker = document.getElementById('theme-color-picker');
    if (colorPicker) {
        colorPicker.addEventListener('input', (e) => {
            const newColor = e.target.value;
            document.documentElement.style.setProperty('--neon-cyan', newColor);
            
            if (window.currentClustersData) {
                const backBtn = document.getElementById('btn-back-macro');
                const isMacro = backBtn ? backBtn.style.display === 'none' : true;
                plotClusters(window.currentClustersData, isMacro);
            }
        });
    }

    const backBtn = document.getElementById('btn-back-macro');
    if (backBtn) {
        backBtn.addEventListener('click', () => {
            if (myMap) {
                myMap.setView([39.8283, -98.5795], 5);
            }
            backBtn.style.display = 'none';

            const selectedCard = document.getElementById('selected-accident-card');
            if (selectedCard) {
                selectedCard.style.display = 'none';
            }

            const titleEl = document.querySelector('.overlay-title');
            const subtitleEl = document.querySelector('.overlay-subtitle');
            if (titleEl) titleEl.textContent = 'GLOBAL INCIDENT OVERVIEW';
            if (subtitleEl) subtitleEl.textContent = 'Real-time traffic accident clusters (3 Classes)';

            if (window.currentClustersData) {
                plotClusters(window.currentClustersData, true);
            }

            if (window.globalShapData) {
                renderShapChart(window.globalShapData);
            }
        });
    }

    const selectAccident = document.getElementById('select-accident-id');
    if (selectAccident) {
        selectAccident.addEventListener('change', async (e) => {
            const accidentId = e.target.value;
            if (!accidentId) return;

            try {
                let detail;
                if (accidentId >= 9990) {
                    detail = window.defaultAccidentsData.find(d => d.id == accidentId);
                } else {
                    const res = await fetch(`http://127.0.0.1:8000/api/acidentes/${accidentId}/?classes=3`);
                    if (!res.ok) throw new Error("Falha ao buscar detalhes do acidente");
                    detail = await res.json();
                }

                if (detail) {
                    const lat = detail.Latitude_Inicial;
                    const lng = detail.Longitude_Inicial;
                    if (myMap && lat && lng) {
                        myMap.setView([lat, lng], 13);
                    }

                    window.updateSelectedAccidentDetails(detail);

                    const backBtn = document.getElementById('btn-back-macro');
                    if (backBtn && backBtn.style.display === 'none') {
                        backBtn.style.display = 'flex';
                        
                        const titleEl = document.querySelector('.overlay-title');
                        const subtitleEl = document.querySelector('.overlay-subtitle');
                        if (titleEl) titleEl.textContent = `CLUSTER ${detail.Cluster_Espacial || 0} DETAILS`;
                        if (subtitleEl) subtitleEl.textContent = `Visualizando acidentes no Cluster ${detail.Cluster_Espacial || 0}`;
                        
                        if (accidentId < 9990) {
                            try {
                                const resClust = await fetch(`http://127.0.0.1:8000/api/acidentes/?limit=100&cluster_id=${detail.Cluster_Espacial}&classes=3`);
                                if (resClust.ok) {
                                    const clustData = await resClust.json();
                                    plotClusters(clustData.results, false);
                                    
                                    const selectEl = document.getElementById('select-accident-id');
                                    if (selectEl) selectEl.value = accidentId;
                                }
                            } catch (clustErr) {
                                console.error(clustErr);
                            }
                        }
                    }
                }
            } catch (err) {
                console.error(err);
            }
        });
    }

    const searchInput = document.getElementById('input-search-id');
    const searchBtn = document.getElementById('btn-search-id');

    async function triggerGlobalSearch() {
        const accidentId = searchInput.value.trim();
        if (!accidentId) {
            alert("Por favor, digite um ID de acidente.");
            return;
        }

        try {
            let detail;
            if (accidentId >= 9990) {
                detail = window.defaultAccidentsData.find(d => d.id == accidentId);
            } else {
                const res = await fetch(`http://127.0.0.1:8000/api/acidentes/${accidentId}/?classes=3`);
                if (!res.ok) {
                    if (res.status === 404) {
                        alert(`Acidente com ID #${accidentId} não foi encontrado no banco de dados.`);
                    } else {
                        throw new Error("Erro na busca");
                    }
                    return;
                }
                detail = await res.json();
            }

            if (detail) {
                const lat = detail.Latitude_Inicial;
                const lng = detail.Longitude_Inicial;
                if (myMap && lat && lng) {
                    myMap.setView([lat, lng], 13);
                }

                window.updateSelectedAccidentDetails(detail);

                const backBtn = document.getElementById('btn-back-macro');
                if (backBtn) backBtn.style.display = 'flex';
                
                const titleEl = document.querySelector('.overlay-title');
                const subtitleEl = document.querySelector('.overlay-subtitle');
                if (titleEl) titleEl.textContent = `CLUSTER ${detail.Cluster_Espacial || 0} DETAILS`;
                if (subtitleEl) subtitleEl.textContent = `Visualizando acidentes no Cluster ${detail.Cluster_Espacial || 0}`;
                
                if (accidentId < 9990) {
                    try {
                        const resClust = await fetch(`http://127.0.0.1:8000/api/acidentes/?limit=100&cluster_id=${detail.Cluster_Espacial}&classes=3`);
                        if (resClust.ok) {
                            const clustData = await resClust.json();
                            plotClusters(clustData.results, false);
                            
                            const selectEl = document.getElementById('select-accident-id');
                            if (selectEl) selectEl.value = accidentId;
                        }
                    } catch (clustErr) {
                        console.error(clustErr);
                    }
                }
            }
        } catch (err) {
            console.error(err);
            alert("Erro ao buscar detalhes do acidente no servidor backend.");
        }
    }

    if (searchBtn && searchInput) {
        searchBtn.addEventListener('click', triggerGlobalSearch);
        searchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                triggerGlobalSearch();
            }
        });
    }

    fetchData();
});

function populateAccidentDropdown(accidents) {
    const selectAccident = document.getElementById('select-accident-id');
    if (selectAccident) {
        selectAccident.innerHTML = '<option value="">Selecione pelo ID...</option>';
        if (accidents && accidents.length > 0) {
            accidents.forEach(point => {
                const opt = document.createElement('option');
                opt.value = point.id;
                opt.textContent = `ID #${point.id} (${getSeverityLabel(point.Severidade)})`;
                selectAccident.appendChild(opt);
            });
        }
    }
}

async function fetchData() {
    try {
        let response;
        let isBackend = false;
        try {
            response = await fetch('http://127.0.0.1:8000/api/dashboard-stats-3classes/');
            if (response.ok) {
                isBackend = true;
            } else {
                throw new Error("API retornou erro");
            }
        } catch (apiErr) {
            console.warn("API Backend de 3 classes offline. Tentando carregar JSON estático de fallback...");
            response = await fetch('dashboard_data_3classes.json');
            if (!response.ok) {
                throw new Error("Não foi possível carregar os dados. Certifique-se de iniciar o Django ou gerar o JSON.");
            }
        }
        
        const data = await response.json();
        
        document.getElementById('kpi-visibility').textContent = data.kpis.avg_visibility.toFixed(2) + " mi";
        document.getElementById('kpi-severity').textContent = "Fatal"; // KPI fixo da severidade máxima
        document.getElementById('kpi-accuracy').textContent = (data.kpis.accuracy * 100).toFixed(1) + "%";

        if (document.getElementById('kpi-active-incidents')) {
            if (isBackend && data.map_clusters) {
                document.getElementById('kpi-active-incidents').textContent = "100.000"; 
                const totalSeverity = data.map_clusters.reduce((acc, c) => acc + c.severity, 0);
                const avgSeverity = data.map_clusters.length > 0 ? (totalSeverity / data.map_clusters.length) : 0;
                document.getElementById('kpi-avg-severity').textContent = avgSeverity.toFixed(2) + "/3";
            } else {
                document.getElementById('kpi-active-incidents').textContent = "2.415"; 
                document.getElementById('kpi-avg-severity').textContent = "2.2/3";
            }
        }
        
        window.globalShapData = data.shap_data;
        renderShapChart(data.shap_data);
        
        renderTimeChart(data.time_data);
        
        if (data.distribution) {
            renderDistributionChart(data.distribution);
        } else {
            renderDistributionChart({ fatal: 25, severe: 139, minor: 68 });
        }
        
        window.currentClustersData = data.map_clusters;
        plotClusters(data.map_clusters, true);

        if (isBackend) {
            try {
                const resAcc = await fetch('http://127.0.0.1:8000/api/acidentes/?limit=100&classes=3');
                if (resAcc.ok) {
                    const accData = await resAcc.json();
                    window.defaultAccidentsData = accData.results;
                    populateAccidentDropdown(accData.results);
                }
            } catch (accErr) {
                console.warn("Falha ao pré-carregar lista de acidentes:", accErr);
            }
        } else {
            window.defaultAccidentsData = [
                { id: 9991, Severidade: 3, Latitude_Inicial: 34.05, Longitude_Inicial: -118.24, Visibilidade_Milhas: 1.2 },
                { id: 9992, Severidade: 1, Latitude_Inicial: 40.71, Longitude_Inicial: -74.00, Visibilidade_Milhas: 10.0 },
                { id: 9993, Severidade: 2, Latitude_Inicial: 41.87, Longitude_Inicial: -87.62, Visibilidade_Milhas: 4.5 },
                { id: 9994, Severidade: 3, Latitude_Inicial: 29.76, Longitude_Inicial: -95.36, Visibilidade_Milhas: 2.0 }
            ];
            populateAccidentDropdown(window.defaultAccidentsData);
        }
        
    } catch (error) {
        console.error(error);
        loadMockData();
    }
}

function loadMockData() {
    console.warn("Usando dados de exemplo (Mock)...");
    
    document.getElementById('kpi-visibility').textContent = "1.2 mi";
    document.getElementById('kpi-accuracy').textContent = "57.9%";
    
    if (document.getElementById('kpi-active-incidents')) {
        document.getElementById('kpi-active-incidents').textContent = "2,415";
        document.getElementById('kpi-avg-severity').textContent = "2.2/3";
    }
    
    renderShapChart({
        labels: ['Visibilidade', 'Hora do Dia', 'Temperatura', 'Precipitação', 'Pressão', 'Umidade'],
        values: [2.9, 2.0, 1.4, 1.2, 0.7, 0.5]
    });
    
    renderTimeChart({
        labels: ['00h', '04h', '08h', '12h', '16h', '20h'],
        low_severity: [50, 30, 200, 150, 250, 100],
        high_severity: [80, 120, 40, 30, 50, 90]
    });
    
    renderDistributionChart({ fatal: 25, severe: 139, minor: 68 });
    
    const mockClusters = [
        { lat: 34.05, lng: -118.24, severity: 3 },
        { lat: 40.71, lng: -74.00, severity: 1 },
        { lat: 41.87, lng: -87.62, severity: 2 },
        { lat: 29.76, lng: -95.36, severity: 3 }
    ];
    window.currentClustersData = mockClusters;
    plotClusters(mockClusters);
}

window.updateSelectedAccidentDetails = function(detail) {
    if (!detail) return;

    const selectedAccidentCard = document.getElementById('selected-accident-card');
    if (selectedAccidentCard) {
        selectedAccidentCard.style.display = 'block';
    }

    const titleEl = document.getElementById('selected-accident-title');
    if (titleEl) {
        titleEl.textContent = `ID: #${detail.id} (Severidade Real: ${getSeverityLabel(detail.Severidade)})`;
    }

    const distEl = document.getElementById('detail-dist');
    if (distEl) {
        const dist = detail.Distancia_Milhas !== undefined ? detail.Distancia_Milhas : 0;
        distEl.textContent = `${dist.toFixed(2)} mi`;
    }

    const rushEl = document.getElementById('detail-rush');
    if (rushEl) {
        rushEl.textContent = detail.Horario_Pico ? 'Sim' : 'Não';
    }

    const crossingEl = document.getElementById('detail-crossing');
    if (crossingEl) {
        crossingEl.textContent = detail.Cruzamento ? 'Sim' : 'Não';
    }

    const signalEl = document.getElementById('detail-signal');
    if (signalEl) {
        signalEl.textContent = detail.Semaforo ? 'Sim' : 'Não';
    }

    const junctionEl = document.getElementById('detail-junction');
    if (junctionEl) {
        junctionEl.textContent = detail.Juncao ? 'Sim' : 'Não';
    }

    const stationEl = document.getElementById('detail-station');
    if (stationEl) {
        stationEl.textContent = detail.Estacao ? 'Sim' : 'Não';
    }

    const predictions = detail.predictions || {};
    
    const predXgbEl = document.getElementById('pred-xgb');
    if (predXgbEl) {
        const val = predictions.xgboost !== undefined ? predictions.xgboost : detail.Severidade;
        predXgbEl.textContent = getSeverityLabel(val);
    }

    const predCnnEl = document.getElementById('pred-cnn');
    if (predCnnEl) {
        const val = predictions.cnn_1d !== undefined ? predictions.cnn_1d : detail.Severidade;
        predCnnEl.textContent = getSeverityLabel(val);
    }

    const predRfEl = document.getElementById('pred-rf');
    if (predRfEl) {
        const val = predictions.random_forest !== undefined ? predictions.random_forest : detail.Severidade;
        predRfEl.textContent = getSeverityLabel(val);
    }

    const predLrEl = document.getElementById('pred-lr');
    if (predLrEl) {
        const val = predictions.logistic_regression !== undefined ? predictions.logistic_regression : detail.Severidade;
        predLrEl.textContent = getSeverityLabel(val);
    }

    const vis = detail.Visibilidade_Milhas !== undefined ? detail.Visibilidade_Milhas : 10.0;
    document.getElementById('kpi-visibility').textContent = `${vis.toFixed(2)} mi`;
    document.getElementById('kpi-severity').textContent = getSeverityLabel(detail.Severidade);

    const tempF = detail.Temperatura_F !== undefined ? detail.Temperatura_F : 60;
    const tempCelsius = Math.round((tempF - 32) * 5 / 9);
    document.getElementById('weather-temp').textContent = `${tempCelsius}°C`;
    
    const hum = detail.Umidade_Percentual !== undefined ? detail.Umidade_Percentual : 50;
    document.getElementById('weather-humidity').textContent = `${Math.round(hum)}%`;
    
    const wind = detail.Velocidade_Vento_Mph !== undefined ? detail.Velocidade_Vento_Mph : 0;
    document.getElementById('weather-wind').textContent = `${Math.round(wind * 1.60934)} km/h`;
    
    let condition = "Clear";
    const prec = detail.Precipitacao_Polegadas !== undefined ? detail.Precipitacao_Polegadas : 0;
    if (prec > 0.05) {
        condition = "Rain";
    } else if (vis < 1.0) {
        condition = "Fog";
    } else if (vis < 4.0) {
        condition = "Overcast";
    } else if (hum > 80) {
        condition = "Cloudy";
    }
    document.getElementById('weather-condition').textContent = condition;

    const avgSevEl = document.getElementById('kpi-avg-severity');
    if (avgSevEl) {
        avgSevEl.textContent = `${detail.Severidade.toFixed(1)}/3`;
    }

    const selectEl = document.getElementById('select-accident-id');
    if (selectEl && detail.id) {
        selectEl.value = detail.id;
    }

    if (detail.shap_values && typeof renderShapChart === 'function') {
        const labels = detail.shap_values.map(v => v.feature);
        const values = detail.shap_values.map(v => v.shap_value);
        renderShapChart({ labels, values });
    } else if (typeof renderShapChart === 'function') {
        renderShapChart({
            labels: ['Visibilidade', 'Cruzamento', 'Semáforo', 'Hora do Dia', 'Temperatura', 'Precipitação'],
            values: [0.35, 0.15, -0.12, 0.28, 0.05, 0.25]
        });
    }
};
