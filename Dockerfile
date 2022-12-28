FROM python

WORKDIR /app

COPY . .

RUN pip install -r /app/requirements.txt

EXPOSE 8050

CMD ["python", "index.py"]
