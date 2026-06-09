/**
 * @file charts_3classes.js
 * @description Script de configuração e renderização dos gráficos interativos para 3 Classes usando Chart.js.
 */

Chart.defaults.color = 'rgba(255, 255, 255, 0.6)';
Chart.defaults.font.family = "'Inter', sans-serif";

let shapChartInstance = null;
let timeChartInstance = null;
let distChartInstance = null;

if (typeof ChartDataLabels !== 'undefined') {
    Chart.register(ChartDataLabels);
}

/**
 * Renderiza o gráfico de barras horizontais do SHAP.
 */
function renderShapChart(data) {
    const ctx = document.getElementById('shapChart').getContext('2d');
    
    if(shapChartInstance) shapChartInstance.destroy();
    
    const gradient = ctx.createLinearGradient(0, 0, 400, 0);
    gradient.addColorStop(0, 'rgba(0, 240, 255, 0.8)');
    gradient.addColorStop(1, 'rgba(59, 130, 246, 0.8)');
    
    shapChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.labels,
            datasets: [{
                data: data.values,
                backgroundColor: gradient,
                borderRadius: 4,
                barThickness: 10
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { 
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
 * Renderiza o gráfico de linhas temporal (24h).
 */
function renderTimeChart(data) {
    const ctx = document.getElementById('timeChart').getContext('2d');
    
    if(timeChartInstance) timeChartInstance.destroy();
    
    const gradCyan = ctx.createLinearGradient(0, 0, 0, 300);
    gradCyan.addColorStop(0, 'rgba(0, 240, 255, 0.4)');
    gradCyan.addColorStop(1, 'rgba(0, 240, 255, 0.0)');
    
    const gradRed = ctx.createLinearGradient(0, 0, 0, 300);
    gradRed.addColorStop(0, 'rgba(255, 42, 85, 0.5)');
    gradRed.addColorStop(1, 'rgba(255, 42, 85, 0.0)');
    
    timeChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.labels,
            datasets: [
                {
                    label: 'Leve/Médio',
                    data: data.low_severity,
                    borderColor: '#00F0FF',
                    backgroundColor: gradCyan,
                    fill: true,
                    tension: 0.5,
                    borderWidth: 2,
                    pointRadius: 0
                },
                {
                    label: 'Grave/Fatal',
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
                mode: 'index',
                intersect: false,
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
                    ticks: { maxTicksLimit: 8 }
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
 * Renderiza o gráfico de distribuição de severidade (3 classes).
 */
function renderDistributionChart(data) {
    const ctx = document.getElementById('distributionChart').getContext('2d');
    
    if(distChartInstance) distChartInstance.destroy();
    
    // Configurações de cores premium para 3 classes:
    // Fatal (Vermelho/Rosa), Grave (Laranja), Leve/Médio (Ciano)
    const colors = [
        { bg: 'rgba(255, 42, 85, 0.4)', border: '#FF2A55', label: '#FB7185' },  // Fatal: Rosa/Vermelho Neon
        { bg: 'rgba(255, 138, 0, 0.4)', border: '#FF8A00', label: '#FBBF24' },  // Grave: Laranja Neon
        { bg: 'rgba(0, 240, 255, 0.4)', border: '#00F0FF', label: '#60A5FA' }   // Leve/Médio: Ciano Neon
    ];

    distChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Fatal', 'Grave', 'Leve/Médio'],
            datasets: [{
                data: [data.fatal, data.severe, data.minor],
                backgroundColor: colors.map(c => c.bg),
                borderColor: colors.map(c => c.border),
                borderWidth: 2,
                borderRadius: 8,
                barThickness: 35
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                datalabels: {
                    anchor: 'end',
                    align: 'top',
                    color: (context) => colors[context.dataIndex].label,
                    font: {
                        family: "'Outfit', sans-serif",
                        weight: 'bold',
                        size: 14
                    },
                    formatter: Math.round
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
                    suggestedMax: Math.max(data.fatal, data.severe, data.minor) * 1.3
                }
            }
        }
    });
}
