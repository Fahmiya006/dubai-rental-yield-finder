FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy
COPY . .
ENV HOST=0.0.0.0 PORT=8000 COOKIE_SECURE=1 RYF_DB=/data/app.db
VOLUME /data
EXPOSE 8000
CMD ["python", "server.py"]
