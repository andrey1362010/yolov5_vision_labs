-----  Обучаем детектор YOLO ------

1. Скачиваем датасет 100 DAYS OF HANDS
http://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/downloads/raw.zip
Распаковываем 
Скачиваем небольшую выборку моей разметки
https://disk.yandex.ru/d/I5C28DqQnCO1-g

2. Подготавливаем данные для обучения
   
   Скрипт boosters/preprocess_yolo_100D.py
   IMAGES_PATH - заменяем на папку 100D/raw датасата 100D
   LABELS_TRAIN_PATH - заменяем на файл /100D/raw/file/trainval.json датасата 100D
   LABELS_TEST_PATH - заменяем на файл /100D/raw/file/test.json датасата 100D
   SAVE_PATH - Указываем куда хотим сохранить обработанные данные

   Скрипт boosters/preprocess_yolo_my_labels.py
   IMAGES_PATH - заменяем на папку train/ датасата VisionLabs
   MY_LABELS_PATH - заменяем на папку со скачанными данными моей разметки
   SAVE_PATH - Указываем куда хотим сохранить обработанные данные
   
3. Обучаем YOLO
    Скрипт train.py
    Параметры: --data boosters/100d_yolo.yaml --batch-size 16 --weights yolov5m.pt --epochs 50 --resume
    Идём пить чай пару дней

-----  Обучаем детектор CLASSIFICATOR ------

4. Подготавливаем данные для обучения классификатора
   Скрипт boosters/crop_detects.py
   WEIGHTS_PATH = Заменяем на путь к обученной модельке yolo должен появиться в папке (/runs/train/exp1/weights/best.pt)
   IMAGES_PATH = Заменяем на шаблон нахождения картинок в датасете VisionLabs, пример: "/media/andrey/big/downloads/train/**/*"
   SAVE_PATH = Куда хотим сохранить обработанные файлы
   
5. Обучаем классификатор
   Скрипт boosters/classificator/train.py
   LABELS_PATH = Путь к "train.csv" датасета VisionLabs
   IMAGES_PATH = Путь к папке сохранённой на 4 этапе
   Пьём чай 20 эпох и выключаем
   В папке boosters/classificator пояфится файл модельки

-----  Сабмитим ------

6. Финаьный скрипт сабмита script.py
   WEIGHTS_PATH - Путь к модельке YOLO
   WEIGHTS_PATH_CLASSIFICATOR - путь к модельке классификатора

