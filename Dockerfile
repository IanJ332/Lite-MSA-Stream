# 使用官方轻量级 Python 镜像
FROM python:3.10-slim as builder

WORKDIR /app

# 安装构建依赖 (如果后续安装 numpy/scipy 需要编译)
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 1. 安装 CPU 版本的 PyTorch (大幅减小体积) 
# 注意：为了 Stage 1 VAD 运行，我们需要 torch 和 torchaudio
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. 安装其他依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- 最终运行镜像 ---
FROM python:3.10-slim

WORKDIR /app

# 安装运行时必要的系统库 (libsndfile 用于音频处理)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 阶段复制安装好的 Python 包
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app ./app
COPY models ./models
COPY frontend ./frontend

# Create logs directory
RUN mkdir logs

# Expose port 8080 (Web + API)
EXPOSE 8080

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
