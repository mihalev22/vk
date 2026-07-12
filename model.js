// ML модель для предсказания параметров улучшения изображений
// Архитектура: 10 признаков → 8 (ReLU) → 3 (sigmoid) → яркость, контраст, цветность

class EnhancementModel {
    constructor() {
        this.model = null;
        this.ready = false;
    }

    async init() {
        if (this.ready) return;

        this.model = tf.sequential({
            layers: [
                tf.layers.dense({ inputShape: [10], units: 8, activation: 'relu' }),
                tf.layers.dense({ units: 3, activation: 'sigmoid' })
            ]
        });

        // tf.js ожидает kernel формы [inputDim, units], а веса ниже записаны
        // построчно "на нейрон" (то есть [units, inputDim]), поэтому их нужно
        // транспонировать перед setWeights — иначе будет ошибка несовпадения
        // формы тензора, и модель вообще не проинициализируется.
        const w1 = tf.tensor2d(this._w1).transpose(); // [10, 8]
        const w2 = tf.tensor2d(this._w2).transpose(); // [8, 3]

        this.model.setWeights([
            w1, tf.tensor1d(this._b1),
            w2, tf.tensor1d(this._b2)
        ]);

        w1.dispose();
        w2.dispose();

        this.ready = true;
    }

    // Извлечение 10 признаков из изображения
    async extractFeatures(imageDataUrl) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                const maxSize = 128;
                const scale = Math.min(maxSize / img.width, maxSize / img.height, 1);
                canvas.width = Math.floor(img.width * scale);
                canvas.height = Math.floor(img.height * scale);
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;

                const n = data.length / 4;
                let sumR = 0, sumG = 0, sumB = 0;
                let sumR2 = 0, sumG2 = 0, sumB2 = 0;
                let minR = 255, minG = 255, minB = 255;
                let maxR = 0, maxG = 0, maxB = 0;

                for (let i = 0; i < data.length; i += 4) {
                    const r = data[i], g = data[i + 1], b = data[i + 2];
                    sumR += r; sumG += g; sumB += b;
                    sumR2 += r * r; sumG2 += g * g; sumB2 += b * b;
                    if (r < minR) minR = r; if (g < minG) minG = g; if (b < minB) minB = b;
                    if (r > maxR) maxR = r; if (g > maxG) maxG = g; if (b > maxB) maxB = b;
                }

                const avgR = sumR / n, avgG = sumG / n, avgB = sumB / n;
                const stdR = Math.sqrt(Math.max(0, sumR2 / n - avgR * avgR));
                const stdG = Math.sqrt(Math.max(0, sumG2 / n - avgG * avgG));
                const stdB = Math.sqrt(Math.max(0, sumB2 / n - avgB * avgB));
                const luminance = 0.299 * avgR + 0.587 * avgG + 0.114 * avgB;

                // Освобождаем Canvas
                canvas.width = 0;
                canvas.height = 0;

                resolve([
                    avgR / 255, avgG / 255, avgB / 255,
                    stdR / 128, stdG / 128, stdB / 128,
                    (maxR - minR) / 255, (maxG - minG) / 255,
                    (minR + minG + minB) / (3 * 255),
                    luminance / 255
                ]);
            };
            img.onerror = () => reject(new Error('Не удалось загрузить изображение'));
            img.src = imageDataUrl;
        });
    }

    // Предсказание параметров улучшения через TF.js модель
    async predict(imageDataUrl) {
        await this.init();
        const features = await this.extractFeatures(imageDataUrl);

        const input = tf.tensor2d([features]);
        const output = this.model.predict(input);
        const raw = Array.from(await output.data());
        input.dispose();
        output.dispose();

        return {
            brightness: 0.6 + raw[0] * 0.8,
            contrast:   0.6 + raw[1] * 0.8,
            saturation: 0.7 + raw[2] * 0.6
        };
    }

    // Применение параметров улучшения к изображению
    // Обработка чанками чтобы не блокировать UI
    async enhance(imageDataUrl, params, outputFormat) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0);

                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                const d = imageData.data;

                const b = params.brightness;
                const c = params.contrast;
                const s = params.saturation;

                const CHUNK_SIZE = 500000; // 500K пикселей за чанк
                let offset = 0;

                function processChunk() {
                    const end = Math.min(offset + CHUNK_SIZE * 4, d.length);

                    for (let i = offset; i < end; i += 4) {
                        let r = d[i] * b;
                        let g = d[i + 1] * b;
                        let bc = d[i + 2] * b;

                        r = c * (r - 128) + 128;
                        g = c * (g - 128) + 128;
                        bc = c * (bc - 128) + 128;

                        const gray = 0.299 * r + 0.587 * g + 0.114 * bc;
                        r = gray + s * (r - gray);
                        g = gray + s * (g - gray);
                        bc = gray + s * (bc - gray);

                        d[i]     = Math.max(0, Math.min(255, r));
                        d[i + 1] = Math.max(0, Math.min(255, g));
                        d[i + 2] = Math.max(0, Math.min(255, bc));
                    }

                    offset = end;

                    if (offset < d.length) {
                        // Уступаем главному потоку для обновления UI
                        setTimeout(processChunk, 0);
                    } else {
                        ctx.putImageData(imageData, 0, 0);

                        // Выбираем формат: PNG для PNG/BMP, JPEG для остальных
                        let mime;
                        if (outputFormat === 'image/png') {
                            mime = 'image/png';
                        } else if (outputFormat === 'image/bmp') {
                            mime = 'image/png'; // BMP не поддерживается canvas.toDataURL
                        } else {
                            mime = 'image/jpeg';
                        }
                        const quality = mime === 'image/jpeg' ? 0.92 : undefined;
                        const result = canvas.toDataURL(mime, quality);

                        // Освобождаем Canvas
                        canvas.width = 0;
                        canvas.height = 0;

                        resolve(result);
                    }
                }

                processChunk();
            };
            img.onerror = () => reject(new Error('Не удалось загрузить изображение для обработки'));
            img.src = imageDataUrl;
        });
    }

    // === Веса модели, закодированные эвристики улучшения ===

    // Слой 1 (10→8, ReLU): детекторы характеристик изображения
    _w1 = [
        [-1,  -1,  -1,   0,   0,   0,   0,   0,   0, -0.5],  // dark
        [ 1,   1,   1,   0,   0,   0,   0,   0,   0,  0.5],  // bright
        [ 0,   0,   0,  -1,  -1,  -1,   0,   0,   0,   0],   // low contrast
        [ 0,   0,   0,   1,   1,   1,   0,   0,   0,   0],   // high contrast
        [ 0,   0,   0,   0,   0,   0,  -1,  -1,   0,   0],   // small DR
        [ 0,   0,   0,   0,   0,   0,   1,   1,   0,   0],   // large DR
        [ 0,   0,   0,   0,   0,   0,   0,   0,  -1,   0],   // dark shadows
        [-0.5,-0.5,-0.5, 0.5, 0.5, 0.5, 0,   0,   0,   0]   // well-exposed
    ];
    _b1 = [1.5, -1.5, 0.6, -0.6, 0.6, -0.6, 0.3, 0.5];

    // Слой 2 (8→3, sigmoid): маппинг детекторов → параметры
    _w2 = [
        [ 0.8, -0.8,   0,   0,   0,   0,  0.3,   0],   // → brightness
        [   0,    0, 0.8,-0.3, 0.4,-0.2,   0,   0],   // → contrast
        [ 0.1,  0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]    // → saturation
    ];
    _b2 = [0, 0, -0.1];
}

window.EnhancementModel = EnhancementModel;