# О сервисе
Веб-сервис `Digital Psycho` позволяет определить эмоции детей и подростков.

# Как запустить бекенд
```bash
docker-compose build app
docker-compose down
docker-compose up -d app
```

# Как запустить модель
## Способ 1
```bash
docker-compose build app
docker-compose down
docker-compose up -d app
docker-compose exec app bash
<положить все видео в папку data внутри репозитория>
python predict_emotions.py
<результаты в файле emotions.csv>
```

## Способ 2
```bash
python3 -m venv venv
. ./venv/bin/activate
pip install -r requirements.txt
<положить все видео в папку data внутри репозитория>
python predict_emotions.py
```
