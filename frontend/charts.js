/**
 * @file charts.js
 * @description Script de configuração e renderização dos gráficos interativos usando a biblioteca Chart.js.
 * Cria visualizações futuristas com gradientes dinâmicos de cor, layouts sem bordas
 * e plugins personalizados para exibição direta de valores numéricos sobre as barras.
 * 
 * Dicas de Inglês (English Tips):
 * - 'Instance' significa instância (objeto ativo de gráfico em execução que precisa ser destruído antes de recriado).
 * - 'Gradient / Linear gradient' refere-se ao gradiente de cores (transição suave entre duas ou mais tonalidades).
 * - 'Tension' especifica a suavidade/curvatura das linhas no gráfico (valores maiores criam curvas mais suaves).
 * - 'Ticks' são as marcas de escala/divisões nos eixos x e y do gráfico.
 * - 'Datalabels' são rótulos de dados adicionados no topo ou dentro das barras/linhas.
 * - 'Border radius' é o arredondamento dos cantos das barras no gráfico.
 */

// Define as fontes globais de texto e cores padrões do Chart.js para se adequar ao design futurista
Chart.defaults.color = 'rgba(255, 255, 255, 0.6)';
Chart.defaults.font.family = "'Inter', sans-serif";

// Variáveis para guardar as instâncias dos gráficos ativos (previne vazamentos de memória e sobreposição)
let shapChartInstance = null;
let timeChartInstance = null;
let distChartInstance = null;

// Registra globalmente o plugin de rótulos de dados (ChartDataLabels) caso a biblioteca tenha sido importada no HTML
if (typeof ChartDataLabels !== 'undefined') {
    Chart.register(ChartDataLabels);
}

/**
 * Renderiza o gráfico de barras horizontais do SHAP (Explicabilidade de Variáveis).
 * 
 * Parâmetros (Parameters):
 * - data (object): Objeto contendo labels (nomes das variáveis) e values (importância média).
 */
function renderShapChart(data) {
    const ctx = document.getElementById('shapChart').getContext('2d');
    
    // Se já existir um gráfico renderizado nesse canvas, destrói a instância anterior
    if(shapChartInstance) shapChartInstance.destroy();
    
    // Cria um gradiente linear horizontal: cor ciano neon para azul royal (horizontal color transition)
    const gradient = ctx.createLinearGradient(0, 0, 400, 0);
    gradient.addColorStop(0, 'rgba(0, 240, 255, 0.8)');
    gradient.addColorStop(1, 'rgba(59, 130, 246, 0.8)');
    
    // Instancia o novo gráfico de barras
    shapChartInstance = new Chart(ctx, {
        type: 'bar',  // Tipo barra
        data: {
            labels: data.labels,
            datasets: [{
                data: data.values,
                backgroundColor: gradient,
                borderRadius: 4,  // Cantos arredondados na extremidade das barras
                barThickness: 10  // Espessura fina para um visual elegante
            }]
        },
        options: {
            indexAxis: 'y',  // Converte em gráfico horizontal (inverte eixos x e y)
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }  // Oculta a legenda já que há apenas uma série de dados
            },
            scales: {
                x: { 
                    // Configura uma grade muito discreta para não poluir o visual escuro (dark HUD layout)
                    grid: { color: 'rgba(255,255,255,0.05)', drawBorder: false },
                    border: { display: false }
                },
                y: { 
                    grid: { display: false },
                    border: { display: false }
                }
            }
        }
    });
}

/**
 * Renderiza o gráfico de linhas temporal mostrando a evolução da severidade de acidentes nas 24h.
 * 
 * Parâmetros (Parameters):
 * - data (object): Contém os rótulos de tempo e vetores das curvas leve (low) e grave (high).
 */
function renderTimeChart(data) {
    const ctx = document.getElementById('timeChart').getContext('2d');
    
    if(timeChartInstance) timeChartInstance.destroy();
    
    // Gradiente Vertical Translúcido de preenchimento para a linha Ciano (Severidades 1 e 2)
    const gradCyan = ctx.createLinearGradient(0, 0, 0, 300);
    gradCyan.addColorStop(0, 'rgba(0, 240, 255, 0.4)');
    gradCyan.addColorStop(1, 'rgba(0, 240, 255, 0.0)');  // Desaparece próximo ao eixo inferior
    
    // Gradiente Vertical Translúcido de preenchimento para a linha Vermelha (Severidades 3 e 4)
    const gradRed = ctx.createLinearGradient(0, 0, 0, 300);
    gradRed.addColorStop(0, 'rgba(255, 42, 85, 0.5)');
    gradRed.addColorStop(1, 'rgba(255, 42, 85, 0.0)');
    
    timeChartInstance = new Chart(ctx, {
        type: 'line',  // Tipo linha
        data: {
            labels: data.labels,
            datasets: [
                {
                    label: 'S1/S2',  // Baixa gravidade (Leve e Moderada)
                    data: data.low_severity,
                    borderColor: '#00F0FF',
                    backgroundColor: gradCyan,
                    fill: true,  // Preenche a área sob a curva
                    tension: 0.5,  // Curvas suaves tipo Spline
                    borderWidth: 2,
                    pointRadius: 0  // Remove os círculos marcadores para uma linha lisa e limpa
                },
                {
                    label: 'G3/G4 Fatal',  // Alta gravidade (Severa e Fatal)
                    data: data.high_severity,
                    borderColor: '#FF2A55',
                    backgroundColor: gradRed,
                    fill: true,
                    tension: 0.5,
                    borderWidth: 2,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',       // Mostra os valores de ambas as curvas simultaneamente no tooltip
                intersect: false,    // Tooltip ativa ao passar o mouse perto do eixo vertical (não exige em cima da linha)
            },
            plugins: {
                legend: {
                    position: 'top',
                    align: 'end',
                    labels: { boxWidth: 10, usePointStyle: true, pointStyle: 'circle' }
                }
            },
            scales: {
                x: { 
                    grid: { display: false }, 
                    border: { display: false }, 
                    ticks: { maxTicksLimit: 8 }  // Evita sobreposição de horários comprimindo a escala horizontal
                },
                y: { 
                    grid: { color: 'rgba(255,255,255,0.05)' }, 
                    border: { display: false } 
                }
            }
        }
    });
}

/**
 * Renderiza o gráfico de barras verticais de distribuição dos incidentes por categoria de risco.
 * Apresenta cores com brilho neon individual por coluna e rótulos numéricos flutuantes.
 * 
 * Parâmetros (Parameters):
 * - data (object): Contém as contagens absolutas para cada nível (fatal, severe, moderate, minor).
 */
function renderDistributionChart(data) {
    const ctx = document.getElementById('distributionChart').getContext('2d');
    
    if(distChartInstance) distChartInstance.destroy();
    
    // Configuração das cores Glow exclusivas de cada barra baseada nas variáveis de design
    const colors = [
        { bg: 'rgba(59, 130, 246, 0.4)', border: '#3B82F6', label: '#60A5FA' }, // Fatal: Azul Neon
        { bg: 'rgba(255, 42, 85, 0.4)', border: '#FF2A55', label: '#FB7185' },  // Severe: Rosa/Vermelho Neon
        { bg: 'rgba(255, 138, 0, 0.4)', border: '#FF8A00', label: '#FBBF24' },  // Moderate: Laranja Neon
        { bg: 'rgba(0, 255, 102, 0.4)', border: '#00FF66', label: '#34D399' }   // Minor: Verde Neon
    ];

    distChartInstance = new Chart(ctx, {
        type: 'bar',  // Tipo barra vertical
        data: {
            labels: ['Fatal', 'Severe', 'Moderate', 'Minor'],
            datasets: [{
                data: [data.fatal, data.severe, data.moderate, data.minor],
                backgroundColor: colors.map(c => c.bg),
                borderColor: colors.map(c => c.border),
                borderWidth: 2,
                borderRadius: 8,  // Cantos superiores das colunas arredondados
                barThickness: 30  // Largura confortável das colunas
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },  // Legenda oculta
                datalabels: {
                    anchor: 'end',           // Posiciona no topo das barras
                    align: 'top',
                    color: (context) => colors[context.dataIndex].label, // Cor correspondente ao tema da barra
                    font: {
                        family: "'Outfit', sans-serif",
                        weight: 'bold',
                        size: 14
                    },
                    formatter: Math.round    // Exibe o número inteiro absoluto arredondado
                }
            },
            scales: {
                x: { 
                    grid: { display: false }, 
                    border: { display: false },
                    ticks: { color: 'rgba(255,255,255,0.7)', font: { size: 12 } }
                },
                y: { 
                    grid: { color: 'rgba(255,255,255,0.05)' }, 
                    border: { display: false },
                    // Sugere um limite máximo extra (suggestedMax) para evitar que o rótulo datalabel bata no teto
                    suggestedMax: Math.max(data.fatal, data.severe, data.moderate, data.minor) * 1.3
                }
            }
        }
    });
}
