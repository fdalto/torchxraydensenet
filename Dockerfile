FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia o arquivo de dependências primeiro (melhor para cache)
COPY requirements.txt .

# Atualiza o apt e instala git (necessário para torchxrayvision)
RUN apt update && apt install -y git && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia o restante dos arquivos do projeto
COPY . .

# Abre o bash para execução interativa
CMD ["/bin/bash"]

