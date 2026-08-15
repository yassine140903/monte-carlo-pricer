FROM python:3.12-slim

WORKDIR /app

# Unbuffered so the seed's progress shows up in `docker-compose logs` as it
# happens rather than in one burst when the process exits.
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY entrypoint.sh .
# The repo is developed on Windows, where the executable bit does not survive;
# set it here so the entrypoint runs regardless of how the file was checked out.
RUN chmod +x entrypoint.sh

EXPOSE 8000

# Seeds market data on first run, then execs uvicorn.
CMD ["./entrypoint.sh"]
