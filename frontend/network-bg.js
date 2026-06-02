/**
 * @file network-bg.js
 * @description Fundo de Tela Interativo de Teia Neural (Neural Network Background Canvas).
 * Renderiza um sistema dinâmico de partículas flutuantes que se conectam por meio de
 * linhas com opacidade variável dependente da proximidade física (distância euclidiana).
 * 
 * Dicas de Inglês (English Tips):
 * - 'Canvas context' é o contexto de renderização gráfica 2D ou 3D sobre o canvas.
 * - 'Particle system' refere-se ao sistema de partículas (agrupamento de pontos matemáticos animados).
 * - 'Velocity / Vector' refere-se à velocidade vetorial dividida nas direções X e Y (vx, vy).
 * - 'Radial gradient' é o gradiente de cores circular (radial) que se espalha do centro para fora.
 * - 'Euclidean distance' significa distância euclidiana (comprimento da linha reta entre dois pontos).
 * - 'Request Animation Frame' é um método do navegador otimizado para executar animações de forma suave e sincronizada com a taxa de atualização do monitor.
 */

// Obtém o elemento canvas e o contexto bidimensional (2D) para desenho de primitivas gráficas
const canvas = document.getElementById('network-bg');
const ctx = canvas.getContext('2d');

let width, height;  // Variáveis para largura e altura físicas da janela do navegador
let particles = []; // Vetor contendo a lista de partículas instanciadas

// Configurações do comportamento dinâmico e estético do sistema de partículas (particle configurations)
const config = {
    particleCount: 150,        // Quantidade total de partículas ativas na tela
    particleBaseRadius: 2,     // Raio máximo base de cada ponto
    lineDistance: 150,         // Distância máxima para gerar uma linha de conexão
    baseSpeed: 0.5,            // Multiplicador da velocidade linear dos pontos
    colors: ['#00d2ff', '#3a7bd5', '#00f0ff'], // Paleta de cores neon aplicável às partículas
    background: '#040b14'      // Cor de fundo padrão (Fallback background)
};

/**
 * Ajusta as dimensões lógicas do canvas para preencher a totalidade da janela do navegador.
 */
function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
}

// Ouve pelo redimensionamento do navegador para recalcular o tamanho do canvas (responsividade)
window.addEventListener('resize', resize);
resize(); // Executa o ajuste inicial de dimensão

/**
 * Representa um ponto físico individual em movimento na tela (Particle Class)
 */
class Particle {
    constructor() {
        // Posicionamento aleatório inicial no espaço bidimensional da tela
        this.x = Math.random() * width;
        this.y = Math.random() * height;
        
        // Vetores de velocidade X e Y gerando direções aleatórias positivas ou negativas (-0.5 a +0.5)
        this.vx = (Math.random() - 0.5) * config.baseSpeed;
        this.vy = (Math.random() - 0.5) * config.baseSpeed;
        
        // Raio aleatório da partícula para que não tenham todas o mesmo tamanho
        this.radius = Math.random() * config.particleBaseRadius + 1;
        
        // Sorteia uma das cores da paleta neon definida nas configurações
        this.color = config.colors[Math.floor(Math.random() * config.colors.length)];
    }

    /**
     * Atualiza a posição espacial somando o vetor de velocidade e rebate nas bordas físicas da tela.
     */
    update() {
        this.x += this.vx;
        this.y += this.vy;

        // Inverte a velocidade horizontal (vx) ao colidir com a borda esquerda ou direita (bounce effect)
        if (this.x < 0 || this.x > width) this.vx *= -1;
        // Inverte a velocidade vertical (vy) ao colidir com a borda superior ou inferior
        if (this.y < 0 || this.y > height) this.vy *= -1;
    }

    /**
     * Renderiza o círculo no Canvas aplicando sombra de desfoque para simular iluminação neon (glow).
     */
    draw() {
        ctx.beginPath();
        // Desenha o arco circular completo (360 graus / 2 * PI radianos)
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fillStyle = this.color;
        
        // Configura propriedades de sombra brilhante
        ctx.shadowBlur = 15;
        ctx.shadowColor = this.color;
        ctx.fill();
        
        // Reseta as propriedades de sombra para não afetar desenhos subsequentes (como as linhas)
        ctx.shadowBlur = 0;
    }
}

/**
 * Instancia a coleção inicial de partículas do sistema.
 */
function initParticles() {
    particles = [];
    for (let i = 0; i < config.particleCount; i++) {
        particles.push(new Particle());
    }
}

/**
 * Calcula a proximidade física entre todas as partículas e traça linhas gradientes de teia neural.
 */
function drawLines() {
    for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
            // Calcula a distância euclidiana por meio da hipotenusa de variação de coordenadas (Pitágoras)
            const dx = particles[i].x - particles[j].x;
            const dy = particles[i].y - particles[j].y;
            const distance = Math.sqrt(dx * dx + dy * dy);

            // Desenha a linha se a distância for menor que o limite máximo (lineDistance)
            if (distance < config.lineDistance) {
                // Opacidade diminui à medida que os pontos se afastam (fade-out effect)
                const opacity = 1 - (distance / config.lineDistance);
                ctx.beginPath();
                ctx.moveTo(particles[i].x, particles[i].y);
                ctx.lineTo(particles[j].x, particles[j].y);
                
                // Cria um gradiente linear ao longo da linha de conexão pintando as pontas com a cor de cada partícula
                const grad = ctx.createLinearGradient(particles[i].x, particles[i].y, particles[j].x, particles[j].y);
                grad.addColorStop(0, particles[i].color);
                grad.addColorStop(1, particles[j].color);
                
                ctx.strokeStyle = `rgba(0, 210, 255, ${opacity * 0.5})`;
                ctx.lineWidth = 1;
                ctx.stroke();
            }
        }
    }
}

/**
 * Laço de animação contínuo (Animation Loop) atualizado via requestAnimationFrame.
 */
function animate() {
    // Desenha o gradiente radial de fundo para criar um efeito cinematográfico de profundidade (radial dark vignetting)
    const bgGrad = ctx.createRadialGradient(width/2, height/2, 0, width/2, height/2, width);
    bgGrad.addColorStop(0, '#0a192f'); // Centro azul escuro translúcido
    bgGrad.addColorStop(1, '#020617'); // Bordas pretas azuladas profundas
    
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, width, height); // Limpa/pinta todo o Canvas

    // Atualiza e desenha cada partícula no frame atual
    for (let p of particles) {
        p.update();
        p.draw();
    }
    
    // Conecta as teias/linhas
    drawLines();
    
    // Solicita ao navegador para rodar novamente o método no próximo ciclo de sincronia de tela
    requestAnimationFrame(animate);
}

// Inicializa a população de partículas e dispara a renderização contínua
initParticles();
animate();
