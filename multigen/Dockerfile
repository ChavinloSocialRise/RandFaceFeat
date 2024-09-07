FROM pytorch/pytorch:latest

WORKDIR /app
COPY .. /app
RUN chmod +x prepare.sh
RUN chmod +x download.sh
RUN ./prepare.sh
RUN ./download.sh

CMD ["python", "-u", "handler.py"]
