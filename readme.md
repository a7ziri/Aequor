<div align="center">
  <img src="img_logo.jpg" alt="Aequor logo" width="250"/>
  <h1>🌟 Aequor</h1>
  <p>Простой и мощный инструмент для обучения  LLM </p>

  [![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
  [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
  [![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](docs/)
  [![Contributors](https://img.shields.io/github/contributors/username/aliment)](https://github.com/username/aliment/graphs/contributors)
</div>

---

## 📚 Содержание
- [О проекте](#-о-проекте)
- [Основные возможности](#-основные-возможности)
- [Быстрый старт](#-быстрый-старт)
- [Установка](#-установка)
- [Использование](#-использование)
- [Архитектура](#-архитектура)
- [Документация](#-документация)
- [План развития](#-план-развития)
- [Участие в проекте](#-участие-в-проекте)
- [Лицензия](#-лицензия)
- [Команда](#-команда)

## 🎯 О проекте

**Aequor** - Это  инструмент для  оубчения или дообучения LLM. Проект разработан для удобства и простоты использования от  разработчика для себя. В будущем будет добавлены почти  все  способы  обучения  llm. DPO OPRO GRPO RM 

### 🌟 Почему Aequor?

- **Гибкость**: Настройка через YAML конфигурации
- **FSDP и  DEEPSPEED**: Оптимизация  обучения как на 1 GPU так и на 1000
- **Масштабируемость**: Модульная архитектура
- **Несколько фреймворков**: HF, Unsloth 

## 🚀 Основные возможности

### 📊 Работа с данными
- Поддержка различных форматов
- Встроенные инструменты визуализации
- Эффективное управление памятью


### 🔄 Callbacks
- Мониторинг процессов
- Логирование
- Кастомные обработчики событий




## 📦 Установка

TODO

## Пример  использования

```bash
# Клонируем репозиторий
git clone https://github.com/username/aliment.git
cd aliment
# Запускаем обучение с использованием Accelerate
accelerate launch --debug --config_file путь_до_конфига путь_до_скрипта  путь_до_конфига_модели
```


## 🔍 Проверка окружения


Для проверки корректности настройки окружения используйте скрипт `env_check.py` в папке `scripts`. 
Успешный результат выполнения должен выглядеть следующим образом:

```bash
CUDA available: True
Number of GPUs: 1 # или больше
_CudaDeviceProperties(name='ваш gpu', major=x, minor=x, total_memory=xxx, multi_processor_count=xxx, L2_cache_size=xx)
2.6.0+cu126    # ваша версия pytorch
12.6           # ваша версия cuda должна соответствовать версии cuda в pytorch
CUDA доступна через Accelerate: True
tensor([0., 0., 0.], device='cuda:0')
```

> **✅ Окружение настроено корректно**, если вы видите вывод `tensor([0., 0., 0.], device='cuda:0')` в конце проверки.

Если  вы  хотите  использовать  другие  способы  обучения  llm  то  вам  нужно  будет  изменить  конфигурационный   config.yaml.
Запустить скрипт  обучения  можно  командой: accelerate launch --debug --config_file путь_к_конфигурационному_файлу путь_к_скрипту_обучения файл_с_конфигурацией_для_обучения

### Структура проекта
src/
├── init.py
├── config.py # Конфигурационные утилиты
├── model_utils.py # Утилиты для работы с моделями
├── yaml_parse.py # Парсер YAML файлов
├── callback/ # Callbacks
├── configs/ # Конфигурационные файлы
├── data/ # Данные проекта
└── scripts/ # Скрипты

## 📈 План развития

### 
- [ ] Расширенная система логирования
- [ ] Добавление  новых  способов  обучения  llm
- [ ] Добавление  новых callbacks 
- [ ] Оптимизация производительности
- [ ] Улучшение документации
- [ ] Web-интерфейс вместо  config.yaml

## 🤝 Участие в проекте
Проект  полностью  открыт  для  вкладов  в  развитие. 
Создаю  его  один  и  хотел  бы  чтобы  он  развивался  вместе  с  сообществом.
Мы приветствуем любой вклад в развитие проекта! 

### Как начать:
1. Форкните репозиторий
2. Создайте ветку для ваших изменений
3. Внесите изменения и создайте pull request

Подробнее в [CONTRIBUTING.md](CONTRIBUTING.md)

## 📝 Документация
TODO

## 🎯 Примеры

### Обученные модели
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-smollm2--360M-yellow)](https://huggingface.co/assskelad/smollm2-360M-sft_SmallThoughts)

Модель [smollm2-360M-sft_SmallThoughts](https://huggingface.co/assskelad/smollm2-360M-sft_SmallThoughts) была успешно обучена с использованием Aequor. 


## 📄 Лицензия

Проект распространяется под лицензией MIT. Подробности в файле [LICENSE](LICENSE)



<div align="center">
  <h3>Built with ❤️ by the Aequor Team</h3>
</div>


