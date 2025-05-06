FROM python:3.12-slim

# Node.js 설치를 위한 의존성 패키지 설치
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p .streamlit
COPY config.toml .streamlit/
COPY . .

RUN npm install -g playwright
RUN npx playwright install chrome

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "application/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
