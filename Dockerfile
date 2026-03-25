FROM registry.linker.cc/linker/bg-base
WORKDIR /app
USER root

COPY . .

ENV LANG C.UTF-8

COPY ./ /app

CMD ["python3", "server.py"]







