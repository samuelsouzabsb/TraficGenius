/**
 * @file app.js
 * @description Script de inicialização e controle principal do Dashboard TraficGenius.
 * Responsável por gerenciar eventos, buscar dados assincronamente da API local (JSON)
 * e delegar as atualizações de interface para os renderizadores do mapa e gráficos.
 * 
 * Dicas de Inglês (English Tips):
 * - 'DOMContentLoaded' é um evento disparado quando o documento HTML inicial é totalmente carregado e analisado.
 * - 'Fetch' significa buscar/obter (método assíncrono padrão para requisições HTTP em JavaScript).
 * - 'Event listener' é um escutador/observador de eventos que executa uma função específica ao detectar uma ação.
 * - 'Theme selector / Color picker' refere-se ao selecionador de cor de tema personalizado.
 * - 'Asynchronous / Async' significa assíncrono (processamento que não bloqueia a execução da thread principal).
 * - 'Fallback / Mock data' são dados de reserva/simulados carregados quando o recurso principal está inacessível.
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Inicializa o mapa geográfico Leaflet (vazio por padrão)
    initMap();

    // 2. Configura o seletor de cores do tema dinâmico (Color Picker)
    const colorPicker = document.getElementById('theme-color-picker');
    if (colorPicker) {
        // Escuta as alterações feitas pelo usuário na cor do seletor
        colorPicker.addEventListener('input', (e) => {
            const newColor = e.target.value;
            // Altera dinamicamente a variável do CSS global no elemento raiz (:root)
            document.documentElement.style.setProperty('--neon-cyan', newColor);
            
            // Re-renderiza os marcadores do mapa para pintar os SVGs com a nova cor selecionada
            if (window.currentClustersData) {
                const backBtn = document.getElementById('btn-back-macro');
                const isMacro = backBtn ? backBtn.style.display === 'none' : true;
                plotClusters(window.currentClustersData, isMacro);
            }
        });
    }

    // 3. Configura o botão de voltar para visão macro
    const backBtn = document.getElementById('btn-back-macro');
    if (backBtn) {
        backBtn.addEventListener('click', () => {
            if (myMap) {
                myMap.setView([39.8283, -98.5795], 5);
            }
            backBtn.style.display = 'none';

            // Oculta o card de detalhes do acidente selecionado
            const selectedCard = document.getElementById('selected-accident-card');
            if (selectedCard) {
                selectedCard.style.display = 'none';
            }

            const titleEl = document.querySelector('.overlay-title');
            const subtitleEl = document.querySelector('.overlay-subtitle');
            if (titleEl) titleEl.textContent = 'GLOBAL INCIDENT OVERVIEW';
            if (subtitleEl) subtitleEl.textContent = 'Real-time traffic accident clusters';

            if (window.currentClustersData) {
                plotClusters(window.currentClustersData, true);
            }

            if (window.globalShapData) {
                renderShapChart(window.globalShapData);
            }
        });
    }

    // 4. Configura o seletor de acidentes individuais por ID
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
                    const res = await fetch(`http://127.0.0.1:8000/api/acidentes/${accidentId}/`);
                    if (!res.ok) throw new Error("Falha ao buscar detalhes do acidente");
                    detail = await res.json();
                }

                if (detail) {
                    // Centraliza o mapa nas coordenadas do acidente e abre um zoom detalhado
                    const lat = detail.Latitude_Inicial;
                    const lng = detail.Longitude_Inicial;
                    if (myMap && lat && lng) {
                        myMap.setView([lat, lng], 13);
                    }

                    // Atualiza a UI com os detalhes do acidente
                    window.updateSelectedAccidentDetails(detail);

                    // Carrega todos os acidentes daquele cluster para plotar no mapa se ainda estiver em modo macro
                    const backBtn = document.getElementById('btn-back-macro');
                    if (backBtn && backBtn.style.display === 'none') {
                        backBtn.style.display = 'flex';
                        
                        const titleEl = document.querySelector('.overlay-title');
                        const subtitleEl = document.querySelector('.overlay-subtitle');
                        if (titleEl) titleEl.textContent = `CLUSTER ${detail.Cluster_Espacial || 0} DETAILS`;
                        if (subtitleEl) subtitleEl.textContent = `Visualizando acidentes no Cluster ${detail.Cluster_Espacial || 0}`;
                        
                        if (accidentId < 9990) {
                            try {
                                const resClust = await fetch(`http://127.0.0.1:8000/api/acidentes/?limit=100&cluster_id=${detail.Cluster_Espacial}`);
                                if (resClust.ok) {
                                    const clustData = await resClust.json();
                                    plotClusters(clustData.results, false);
                                    
                                    // Re-seleciona no dropdown novo populado
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

    // 5. Configura a busca global de acidentes por ID digitado
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
                const res = await fetch(`http://127.0.0.1:8000/api/acidentes/${accidentId}/`);
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
                // Centraliza o mapa nas coordenadas do acidente e abre um zoom detalhado
                const lat = detail.Latitude_Inicial;
                const lng = detail.Longitude_Inicial;
                if (myMap && lat && lng) {
                    myMap.setView([lat, lng], 13);
                }

                // Atualiza a UI com os detalhes do acidente
                window.updateSelectedAccidentDetails(detail);

                // Carrega todos os acidentes daquele cluster para plotar no mapa
                const backBtn = document.getElementById('btn-back-macro');
                if (backBtn) backBtn.style.display = 'flex';
                
                const titleEl = document.querySelector('.overlay-title');
                const subtitleEl = document.querySelector('.overlay-subtitle');
                if (titleEl) titleEl.textContent = `CLUSTER ${detail.Cluster_Espacial || 0} DETAILS`;
                if (subtitleEl) subtitleEl.textContent = `Visualizando acidentes no Cluster ${detail.Cluster_Espacial || 0}`;
                
                if (accidentId < 9990) {
                    try {
                        const resClust = await fetch(`http://127.0.0.1:8000/api/acidentes/?limit=100&cluster_id=${detail.Cluster_Espacial}`);
                        if (resClust.ok) {
                            const clustData = await resClust.json();
                            plotClusters(clustData.results, false);
                            
                            // Seleciona o ID no dropdown
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

    // 6. Executa a requisição assíncrona para buscar os dados de visualização
    fetchData();
});

/**
 * Popula o dropdown selector com os acidentes fornecidos.
 */
function populateAccidentDropdown(accidents) {
    const selectAccident = document.getElementById('select-accident-id');
    if (selectAccident) {
        selectAccident.innerHTML = '<option value="">Selecione pelo ID...</option>';
        if (accidents && accidents.length > 0) {
            accidents.forEach(point => {
                const opt = document.createElement('option');
                opt.value = point.id;
                opt.textContent = `ID #${point.id} (Severidade G${point.Severidade})`;
                selectAccident.appendChild(opt);
            });
        }
    }
}

/**
 * Busca os dados agregados gerados no pipeline Python por meio de requisição assíncrona.
 * Atualiza KPIs e aciona a renderização de gráficos e do mapa.
 */
async function fetchData() {
    try {
        let response;
        let isBackend = false;
        try {
            // Tenta chamar a API Django local primeiro
            response = await fetch('http://127.0.0.1:8000/api/dashboard-stats/');
            if (response.ok) {
                isBackend = true;
            } else {
                throw new Error("API retornou erro");
            }
        } catch (apiErr) {
            console.warn("API Backend offline ou inacessível. Tentando carregar JSON estático de fallback...");
            response = await fetch('dashboard_data.json');
            if (!response.ok) {
                throw new Error("Não foi possível carregar os dados. Certifique-se de iniciar o Django ou gerar o JSON.");
            }
        }
        
        // Converte a resposta bruta em objeto JSON
        const data = await response.json();
        
        // --- Atualização dos Indicadores Numéricos (KPIs - Right Side HUD) ---
        document.getElementById('kpi-visibility').textContent = data.kpis.avg_visibility.toFixed(2) + " mi";
        document.getElementById('kpi-severity').textContent = "G" + data.kpis.max_severity;
        document.getElementById('kpi-accuracy').textContent = (data.kpis.accuracy * 100).toFixed(1) + "%";

        // --- Atualização dos KPIs Dinâmicos de Sobreposição (Floating Map Overlay) ---
        if (document.getElementById('kpi-active-incidents')) {
            if (isBackend && data.map_clusters) {
                document.getElementById('kpi-active-incidents').textContent = "100.000"; 
                const totalSeverity = data.map_clusters.reduce((acc, c) => acc + c.severity, 0);
                const avgSeverity = data.map_clusters.length > 0 ? (totalSeverity / data.map_clusters.length) : 0;
                document.getElementById('kpi-avg-severity').textContent = avgSeverity.toFixed(2) + "/4";
            } else {
                document.getElementById('kpi-active-incidents').textContent = "2.415"; 
                document.getElementById('kpi-avg-severity').textContent = "3.8/5";
            }
        }
        
        // --- Renderização Gráfica (Charts Update) ---
        window.globalShapData = data.shap_data;
        renderShapChart(data.shap_data);
        
        // Séries temporais de severidade por hora (timeChart)
        renderTimeChart(data.time_data);
        
        // Distribuição geral de ocorrências de acidentes (distributionChart)
        if (data.distribution) {
            renderDistributionChart(data.distribution);
        } else {
            renderDistributionChart({ fatal: 25, severe: 139, moderate: 50, minor: 18 });
        }
        
        // --- Plotagem de Coordenadas Geográficas (Map Rendering) ---
        window.currentClustersData = data.map_clusters;
        plotClusters(data.map_clusters, true);

        // --- Pré-carrega a lista inicial de acidentes para o dropdown (se a API estiver online) ---
        if (isBackend) {
            try {
                const resAcc = await fetch('http://127.0.0.1:8000/api/acidentes/?limit=100');
                if (resAcc.ok) {
                    const accData = await resAcc.json();
                    window.defaultAccidentsData = accData.results;
                    populateAccidentDropdown(accData.results);
                }
            } catch (accErr) {
                console.warn("Falha ao pré-carregar lista de acidentes:", accErr);
            }
        } else {
            // Em mock, populamos com alguns dados simulados correspondentes aos mockClusters
            window.defaultAccidentsData = [
                { id: 9991, Severidade: 4, Latitude_Inicial: 34.05, Longitude_Inicial: -118.24, Visibilidade_Milhas: 1.2 },
                { id: 9992, Severidade: 2, Latitude_Inicial: 40.71, Longitude_Inicial: -74.00, Visibilidade_Milhas: 10.0 },
                { id: 9993, Severidade: 3, Latitude_Inicial: 41.87, Longitude_Inicial: -87.62, Visibilidade_Milhas: 4.5 },
                { id: 9994, Severidade: 4, Latitude_Inicial: 29.76, Longitude_Inicial: -95.36, Visibilidade_Milhas: 2.0 }
            ];
            populateAccidentDropdown(window.defaultAccidentsData);
        }
        
    } catch (error) {
        console.error(error);
        loadMockData();
    }
}

/**
 * Carrega e injeta dados de teste/demonstração simulados (Mock datasets) no painel.
 * Garante que a interface do usuário seja populada mesmo em ausência de conexões ou pipelines locais.
 */
function loadMockData() {
    console.warn("Usando dados de exemplo (Mock)...");
    
    // Injeta textos simulados nos KPIs correspondentes
    document.getElementById('kpi-visibility').textContent = "1.2 mi";
    document.getElementById('kpi-accuracy').textContent = "89.4%";
    
    if (document.getElementById('kpi-active-incidents')) {
        document.getElementById('kpi-active-incidents').textContent = "2,415";
        document.getElementById('kpi-avg-severity').textContent = "3.8/5";
    }
    
    // Renderiza dados simulados no gráfico de importância SHAP
    renderShapChart({
        labels: ['Visibilidade', 'Hora do Dia', 'Temperatura', 'Precipitação', 'Pressão', 'Umidade'],
        values: [2.5, 1.8, 1.2, 0.9, 0.5, 0.3]
    });
    
    // Renderiza curvas simuladas no gráfico temporal
    renderTimeChart({
        labels: ['00h', '04h', '08h', '12h', '16h', '20h'],
        low_severity: [50, 30, 200, 150, 250, 100],
        high_severity: [80, 120, 40, 30, 50, 90]
    });
    
    // Renderiza gráfico de barras simulando a distribuição
    renderDistributionChart({ fatal: 25, severe: 139, moderate: 50, minor: 18 });
    
    // Lista simulada de coordenadas de acidentes geográficos para o mapa
    const mockClusters = [
        { lat: 34.05, lng: -118.24, severity: 4 }, // Los Angeles, CA
        { lat: 40.71, lng: -74.00, severity: 2 },  // New York, NY
        { lat: 41.87, lng: -87.62, severity: 3 },  // Chicago, IL
        { lat: 29.76, lng: -95.36, severity: 4 }   // Houston, TX
    ];
    window.currentClustersData = mockClusters;
    plotClusters(mockClusters);
}

/**
 * Atualiza o card lateral de detalhes do acidente selecionado, clima, KPIs e predições multiclasse.
 * @param {object} detail Objeto contendo todas as variáveis reais e simuladas do acidente.
 */
window.updateSelectedAccidentDetails = function(detail) {
    if (!detail) return;

    // 1. Exibe o painel de detalhes do acidente selecionado
    const selectedAccidentCard = document.getElementById('selected-accident-card');
    if (selectedAccidentCard) {
        selectedAccidentCard.style.display = 'block';
    }

    // 2. Popula informações textuais do acidente
    const titleEl = document.getElementById('selected-accident-title');
    if (titleEl) {
        titleEl.textContent = `ID: #${detail.id} (Severidade Real: G${detail.Severidade})`;
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

    // 3. Popula predições individuais para cada um dos 4 modelos
    const predictions = detail.predictions || {};
    
    const predXgbEl = document.getElementById('pred-xgb');
    if (predXgbEl) {
        const val = predictions.xgboost !== undefined ? predictions.xgboost : detail.Severidade;
        predXgbEl.textContent = `Grau ${val}`;
    }

    const predCnnEl = document.getElementById('pred-cnn');
    if (predCnnEl) {
        const val = predictions.cnn_1d !== undefined ? predictions.cnn_1d : detail.Severidade;
        predCnnEl.textContent = `Grau ${val}`;
    }

    const predRfEl = document.getElementById('pred-rf');
    if (predRfEl) {
        const val = predictions.random_forest !== undefined ? predictions.random_forest : detail.Severidade;
        predRfEl.textContent = `Grau ${val}`;
    }

    const predLrEl = document.getElementById('pred-lr');
    if (predLrEl) {
        const val = predictions.logistic_regression !== undefined ? predictions.logistic_regression : detail.Severidade;
        predLrEl.textContent = `Grau ${val}`;
    }

    // 4. Atualiza os KPIs HUD rápidos
    const vis = detail.Visibilidade_Milhas !== undefined ? detail.Visibilidade_Milhas : 10.0;
    document.getElementById('kpi-visibility').textContent = `${vis.toFixed(2)} mi`;
    document.getElementById('kpi-severity').textContent = `G${detail.Severidade}`;

    // 5. Atualiza Clima em tempo real no HUD
    const tempF = detail.Temperatura_F !== undefined ? detail.Temperatura_F : 60;
    const tempCelsius = Math.round((tempF - 32) * 5 / 9);
    document.getElementById('weather-temp').textContent = `${tempCelsius}°C`;
    
    const hum = detail.Umidade_Percentual !== undefined ? detail.Umidade_Percentual : 50;
    document.getElementById('weather-humidity').textContent = `${Math.round(hum)}%`;
    
    const wind = detail.Velocidade_Vento_Mph !== undefined ? detail.Velocidade_Vento_Mph : 0;
    document.getElementById('weather-wind').textContent = `${Math.round(wind * 1.60934)} km/h`;
    
    // Determinar condição climática dinamicamente
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

    // 6. Atualiza KPIs de sobreposição do mapa (Overlay panel)
    const avgSevEl = document.getElementById('kpi-avg-severity');
    if (avgSevEl) {
        avgSevEl.textContent = `${detail.Severidade.toFixed(1)}/4`;
    }

    // 7. Sincroniza o seletor dropdown principal (Dropdown selector synchronization)
    const selectEl = document.getElementById('select-accident-id');
    if (selectEl && detail.id) {
        selectEl.value = detail.id;
    }

    // 8. Desenha o gráfico SHAP local deste acidente específico
    if (detail.shap_values && typeof renderShapChart === 'function') {
        const labels = detail.shap_values.map(v => v.feature);
        const values = detail.shap_values.map(v => v.shap_value);
        renderShapChart({ labels, values });
    } else if (typeof renderShapChart === 'function') {
        // Fallback SHAP local caso a API falhe ou não tenha SHAP detalhado
        renderShapChart({
            labels: ['Visibilidade', 'Cruzamento', 'Semáforo', 'Hora do Dia', 'Temperatura', 'Precipitação'],
            values: [0.35, 0.15, -0.12, 0.28, 0.05, 0.25]
        });
    }
};
