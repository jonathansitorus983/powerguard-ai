FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

RUN python app/generate_data.py && python app/train_model.py

EXPOSE 8501 8000
CMD ["python", "-m", "streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0"]
