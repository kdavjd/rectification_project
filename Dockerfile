# Dockerfile
# https://docker.github.io/engine/reference/builder/

FROM python:3.11

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN set -eux; \
  pip install gunicorn \
  pip install --no-deps -r /app/requirements.txt

COPY . .
COPY entrypoint.sh /opt/entrypoint.sh

CMD ["/opt/entrypoint.sh"]
