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
                plotClusters(window.currentClustersData);
            }
        });
    }

    // 3. Executa a requisição assíncrona para buscar os dados de visualização
    fetchData();
});

/**
 * Busca os dados agregados gerados no pipeline Python por meio de requisição assíncrona.
 * Atualiza KPIs e aciona a renderização de gráficos e do mapa.
 */
async function fetchData() {
    try {
        // Realiza a chamada HTTP local para obter o arquivo JSON de dados consolidado
        const response = await fetch('dashboard_data.json');
        
        // Verifica se a resposta HTTP retornou status de sucesso (código 200-299)
        if (!response.ok) {
            throw new Error("Não foi possível carregar os dados. Você executou o script export_frontend_data.py?");
        }
        
        // Converte a resposta bruta (raw response) em um objeto literal JavaScript (JSON parsing)
        const data = await response.json();
        
        // --- Atualização dos Indicadores Numéricos (KPIs - Right Side HUD) ---
        document.getElementById('kpi-visibility').textContent = data.kpis.avg_visibility.toFixed(2) + " mi";
        document.getElementById('kpi-severity').textContent = "G" + data.kpis.max_severity;
        document.getElementById('kpi-accuracy').textContent = (data.kpis.accuracy * 100).toFixed(1) + "%";

        // --- Atualização dos KPIs Dinâmicos de Sobreposição (Floating Map Overlay) ---
        if (document.getElementById('kpi-active-incidents')) {
            document.getElementById('kpi-active-incidents').textContent = "2,415"; // Valor fixo baseado no design padrão
            document.getElementById('kpi-avg-severity').textContent = "3.8/5";
        }
        
        // --- Renderização Gráfica (Charts Update) ---
        // Explicabilidade de variáveis SHAP (shapChart)
        renderShapChart(data.shap_data);
        // Séries temporais de severidade por hora (timeChart)
        renderTimeChart(data.time_data);
        
        // Distribuição geral de ocorrências de acidentes (distributionChart)
        if (data.distribution) {
            renderDistributionChart(data.distribution);
        } else {
            // Mock de distribuição de severidade caso os dados do pipeline ainda não contenham essa chave
            renderDistributionChart({ fatal: 25, severe: 139, moderate: 50, minor: 18 });
        }
        
        // --- Plotagem de Coordenadas Geográficas (Map Rendering) ---
        // Salva os pontos carregados em uma variável global para reutilização nas mudanças de tema
        window.currentClustersData = data.map_clusters;
        plotClusters(data.map_clusters);
        
    } catch (error) {
        console.error(error);
        // Fallback: Se o JSON de dados não for localizado, carrega o painel com dados simulados
        // para manter a visualização rica e interativa para demonstração
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
