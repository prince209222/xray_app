FROM python:3.9-slim

WORKDIR /app

COPY ./pkg_cache/ /tmp/pkgs/
RUN pip install --no-index --find-links=/tmp/pkgs/ -r requirements.txt

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
