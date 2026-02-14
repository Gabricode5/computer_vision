FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

COPY project_pose/requirement.txt /app/requirements.txt

RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "project_pose.agent:app", "--host", "0.0.0.0", "--port", "8000"]
