FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -U pip && pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit","run","apps/dashboard/Home.py","--server.port=8501","--server.address=0.0.0.0"]
