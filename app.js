// ===== API для работы с улучшением изображений =====

class ImageEnhancementAPI {
    constructor() {
        this.model = new EnhancementModel();
        this.tasks = new Map();
        this.counter = 0;
        this.listeners = [];
    }

    // Метод постановки задачи — принимает изображение и формат, возвращает ID задачи
    createTask(imageDataUrl, format) {
        const id = `task_${Date.now()}_${++this.counter}`;
        this.tasks.set(id, {
            id,
            status: 'pending',
            progress: 0,
            input: imageDataUrl,
            result: null,
            params: null,
            format: format || 'image/jpeg',
            error: null
        });
        this._process(id);
        return id;
    }

    // Метод получения статуса задачи — возвращает статус и прогресс
    getTaskStatus(taskId) {
        const task = this.tasks.get(taskId);
        if (!task) throw new Error(`Задача ${taskId} не найдена`);
        return { id: task.id, status: task.status, progress: task.progress };
    }

    // Метод прерывания задачи — возвращает true если успешно
    cancelTask(taskId) {
        const task = this.tasks.get(taskId);
        if (!task) throw new Error(`Задача ${taskId} не найдена`);
        if (['completed', 'failed', 'cancelled'].includes(task.status)) return false;
        task.status = 'cancelled';
        this._emit(taskId, task.status, task.progress);
        return true;
    }

    // Метод получения готового изображения — возвращает dataUrl и параметры
    getResult(taskId) {
        const task = this.tasks.get(taskId);
        if (!task) throw new Error(`Задача ${taskId} не найдена`);
        if (task.status !== 'completed') throw new Error(`Задача не завершена: ${task.status}`);
        return { dataUrl: task.result, params: task.params, format: task.format };
    }

    // Событие изменения статуса задачи
    onStatusChange(callback) {
        this.listeners.push(callback);
    }

    _emit(taskId, status, progress) {
        this.listeners.forEach(fn => fn({ taskId, status, progress }));
    }

    // Удаление завершённой задачи из памяти
    cleanupTask(taskId) {
        this.tasks.delete(taskId);
    }

    // Асинхронная обработка задачи (не блокирует UI)
    async _process(taskId) {
        const task = this.tasks.get(taskId);
        if (!task) return;

        try {
            task.status = 'processing';
            this._emit(taskId, 'processing', 5);

            // Загрузка ML-модели (один раз)
            this._emit(taskId, 'processing', 10);
            await this.model.init();
            if (task.status === 'cancelled') return;

            // Анализ изображения и предсказание параметров
            this._emit(taskId, 'processing', 30);
            const params = await this.model.predict(task.input);
            if (task.status === 'cancelled') return;

            task.params = params;
            this._emit(taskId, 'processing', 50);

            // Применение улучшений к изображению
            this._emit(taskId, 'processing', 60);
            const result = await this.model.enhance(task.input, params, task.format);
            if (task.status === 'cancelled') return;

            task.result = result;
            this._emit(taskId, 'processing', 95);

            task.status = 'completed';
            task.progress = 100;
            this._emit(taskId, 'completed', 100);
        } catch (err) {
            task.status = 'failed';
            task.error = err.message;
            this._emit(taskId, 'failed', 0);
        }
    }
}

// ===== Интерфейс пользователя =====

class App {
    constructor() {
        this.api = new ImageEnhancementAPI();
        this.taskId = null;
        this.originalData = null;
        this.enhancedData = null;
        this.mlParams = null;
        this.inputFormat = null;
        this._sliderTimer = null;

        this.$ = {
            uploadArea: document.getElementById('uploadArea'),
            fileInput: document.getElementById('fileInput'),
            progressWrap: document.getElementById('progressContainer'),
            progressBar: document.getElementById('progress'),
            status: document.getElementById('status'),
            result: document.getElementById('result'),
            originalImg: document.getElementById('originalImage'),
            enhancedImg: document.getElementById('enhancedImage'),
            downloadBtn: document.getElementById('downloadBtn'),
            resetBtn: document.getElementById('resetBtn'),
            cancelBtn: document.getElementById('cancelBtn'),
            brightnessSlider: document.getElementById('brightnessSlider'),
            contrastSlider: document.getElementById('contrastSlider'),
            saturationSlider: document.getElementById('saturationSlider')
        };

        this._bind();
    }

    _bind() {
        const { uploadArea, fileInput, downloadBtn, resetBtn, cancelBtn } = this.$;

        uploadArea.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', e => {
            if (e.target.files.length) this._handleFile(e.target.files[0]);
        });

        uploadArea.addEventListener('dragover', e => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
        uploadArea.addEventListener('drop', e => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files.length) this._handleFile(e.dataTransfer.files[0]);
        });

        downloadBtn.addEventListener('click', () => this._download());
        resetBtn.addEventListener('click', () => this._reset());
        cancelBtn.addEventListener('click', () => this._cancel());

        ['brightnessSlider', 'contrastSlider', 'saturationSlider'].forEach(key => {
            this.$[key].addEventListener('input', () => this._debouncedSlider());
        });

        this.api.onStatusChange(detail => this._onStatus(detail));
    }

    async _handleFile(file) {
        // Определяем формат по расширению и MIME
        const name = file.name.toLowerCase();
        let format = 'image/jpeg';

        if (name.endsWith('.png') || file.type === 'image/png') {
            format = 'image/png';
        } else if (name.endsWith('.bmp') || file.type === 'image/bmp' ||
                   file.type === 'image/x-ms-bmp' || file.type === 'image/x-bmp') {
            format = 'image/bmp';
        } else if (name.endsWith('.heic') || name.endsWith('.heif') ||
                   file.type === 'image/heic' || file.type === 'image/heif') {
            format = 'image/jpeg';
        }

        // Конвертация HEIC → JPEG
        if (format === 'image/jpeg' && (name.endsWith('.heic') || name.endsWith('.heif') ||
            file.type === 'image/heic' || file.type === 'image/heif')) {
            if (typeof heic2any === 'undefined') {
                alert('Библиотека конвертации HEIC не загружена. Проверьте подключение к интернету.');
                return;
            }
            try {
                const blob = await heic2any({ blob: file, toType: 'image/jpeg', quality: 0.92 });
                file = blob;
            } catch (err) {
                alert('Ошибка конвертации HEIC: ' + err.message);
                return;
            }
        }

        const reader = new FileReader();
        reader.onload = e => {
            this.originalData = e.target.result;
            this.inputFormat = format;

            const img = new Image();
            img.onload = () => {
                const megapixels = (img.width * img.height) / 1_000_000;
                if (megapixels > 15) {
                    alert(`Изображение слишком большое: ${megapixels.toFixed(1)} Мпк. Максимум — 15 Мпк.`);
                    this._reset();
                    return;
                }
                this._startProcessing();
            };
            img.onerror = () => {
                alert('Не удалось прочитать изображение');
                this._reset();
            };
            img.src = this.originalData;
        };
        reader.readAsDataURL(file);
    }

    _startProcessing() {
        // Отменяем предыдущую задачу, если она ещё выполняется
        if (this.taskId) {
            try { this.api.cancelTask(this.taskId); } catch (e) {}
        }

        this.$.originalImg.src = this.originalData;
        this.$.uploadArea.style.display = 'none';
        this.$.progressWrap.style.display = 'block';
        this.$.result.style.display = 'none';
        this.$.cancelBtn.style.display = 'inline-block';

        // Формат передаётся напрямую в createTask — без гонки данных
        this.taskId = this.api.createTask(this.originalData, this.inputFormat);
    }

    _onStatus({ taskId, status, progress }) {
        if (taskId !== this.taskId) return;

        this.$.progressBar.style.width = progress + '%';

        if (status === 'processing') {
            this.$.status.textContent = `Обработка... ${progress}%`;
        } else if (status === 'completed') {
            const res = this.api.getResult(this.taskId);
            this.enhancedData = res.dataUrl;
            this.mlParams = res.params;

            this.$.brightnessSlider.value = Math.round(res.params.brightness * 100);
            this.$.contrastSlider.value = Math.round(res.params.contrast * 100);
            this.$.saturationSlider.value = Math.round(res.params.saturation * 100);

            this.$.enhancedImg.src = this.enhancedData;
            this.$.progressWrap.style.display = 'none';
            this.$.result.style.display = 'block';
            this.$.cancelBtn.style.display = 'none';
        } else if (status === 'failed') {
            this.$.status.textContent = 'Ошибка обработки';
            this.$.cancelBtn.style.display = 'none';
            setTimeout(() => this._reset(), 2000);
        } else if (status === 'cancelled') {
            this.$.status.textContent = 'Отменено';
            this.$.cancelBtn.style.display = 'none';
            setTimeout(() => this._reset(), 1500);
        }
    }

    async _onSliderChange() {
        if (!this.originalData) return;

        const params = {
            brightness: this.$.brightnessSlider.value / 100,
            contrast: this.$.contrastSlider.value / 100,
            saturation: this.$.saturationSlider.value / 100
        };

        const result = await this.api.model.enhance(this.originalData, params, this.inputFormat);
        this.enhancedData = result;
        this.$.enhancedImg.src = result;
    }

    _debouncedSlider() {
        clearTimeout(this._sliderTimer);
        this._sliderTimer = setTimeout(() => this._onSliderChange(), 150);
    }

    _cancel() {
        if (this.taskId) {
            this.api.cancelTask(this.taskId);
        }
    }

    _download() {
        if (!this.enhancedData) return;
        const a = document.createElement('a');
        a.href = this.enhancedData;
        // BMP на выходе тоже сохраняется как PNG (canvas.toDataURL не умеет в BMP),
        // поэтому расширение файла должно учитывать оба случая, иначе получится
        // PNG-файл с расширением .jpg
        const ext = (this.inputFormat === 'image/png' || this.inputFormat === 'image/bmp') ? 'png' : 'jpg';
        a.download = `enhanced_${Date.now()}.${ext}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    _reset() {
        // Очищаем старые данные из памяти
        if (this.taskId) {
            this.api.cleanupTask(this.taskId);
        }

        this.taskId = null;
        this.originalData = null;
        this.enhancedData = null;
        this.mlParams = null;
        this.inputFormat = null;
        this.$.fileInput.value = '';

        this.$.uploadArea.style.display = '';
        this.$.progressWrap.style.display = 'none';
        this.$.result.style.display = 'none';
        this.$.progressBar.style.width = '0%';
        this.$.status.textContent = 'Загрузка модели...';
        this.$.cancelBtn.style.display = 'none';

        this.$.brightnessSlider.value = 100;
        this.$.contrastSlider.value = 100;
        this.$.saturationSlider.value = 100;
    }
}

document.addEventListener('DOMContentLoaded', () => new App());