async function carregarPacientes() {
    const container = document.getElementById('container');

    try {
        // Lê o arquivo de lista
        const listaResp = await fetch('output/list.txt');
        const conteudoLista = await listaResp.text();
        const arquivos = conteudoLista.trim().split('\n');

        for (let nome of arquivos) {
            const an = nome.replace('.csv', '').trim();
            const csvUrl = `output/${nome}`;
            const originalUrl = `output/${an}_original.jpg`;
            const mapaUrl = `output/${an}_mapa.jpg`;

            const respostaCSV = await fetch(csvUrl);
            const texto = await respostaCSV.text();
            const linhas = texto.trim().split('\n');
            let normal = true;
            let resultadoTexto = "";

            for (let linha of linhas) {
                const [label, probabilidade] = linha.split(',');
                const valor = parseFloat(probabilidade);
                resultadoTexto += `${label.trim()}: ${valor.toFixed(2)}\n`;
                if (valor >= 0.6) normal = false;
            }

            const div = document.createElement('div');
            div.className = 'paciente';

            const imgOriginal = document.createElement('img');
            imgOriginal.src = originalUrl;
            imgOriginal.className = 'img-exame';

            const imgMapa = document.createElement('img');
            imgMapa.src = mapaUrl;
            imgMapa.className = 'img-mapa';

            const info = document.createElement('div');
            info.className = 'info';

            const titulo = document.createElement('div');
            titulo.className = normal ? 'titulo-normal' : 'titulo-alterado';
            titulo.textContent = normal ? 'Exame normal' : 'Exame alterado';

            const resultado = document.createElement('div');
            resultado.className = 'resultado';
            resultado.textContent = resultadoTexto;
            resultado.style.position = 'relative'; // garante ancoragem do botão

            const botao = document.createElement('button');
            botao.className = 'botao-errou';
            botao.onclick = () => gravar_errou();

            const imgBotao = document.createElement('img');
            imgBotao.src = 'ErrouFausto.svg';
            imgBotao.alt = 'Errou!';
            imgBotao.className = 'img-errou';

            botao.appendChild(imgBotao);
            resultado.appendChild(botao);

            // montar árvore
            info.appendChild(titulo);
            info.appendChild(resultado);

            div.appendChild(imgOriginal);
            div.appendChild(info);
            div.appendChild(imgMapa);

            container.appendChild(div);
        }

    } catch (e) {
        console.error('Erro ao carregar arquivos:', e);
    }
}
window.onload = carregarPacientes;
