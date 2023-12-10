FROM pytorch/pytorch:latest

ENV PYTHONUNBUFFERED=1
ARG TZ=Asia/Tokyo

WORKDIR /app
COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

COPY app.py app.py
COPY vectorize.py vectorize.py
COPY util.py util.py

RUN mkdir -p ~/.streamlit/ ; echo "[general]"  > ~/.streamlit/credentials.toml ; echo "email = \"\""  >> ~/.streamlit/credentials.toml

CMD ["python", "vectorize.py"]
