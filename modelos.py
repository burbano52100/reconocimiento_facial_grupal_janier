"""Ejecuta los tres modelos de Teachable Machine usando solo numpy.

- ClasificadorImagen: modelo Keras (.h5) = MobileNetV2 + capas densas.
- ClasificadorPose:   modelo TF.js de pose = PoseNet (MobileNetV1) + capas densas.
- ClasificadorVoz:    modelo TF.js de audio = espectrograma + red convolucional (speech-commands).
- DetectorFrase:      recorta cada frase hablada (voz entre dos silencios).
- AnalizadorCaras:    detector de caras (YuNet) + huella facial (SFace) de OpenCV.
"""
import base64
import collections
import json
import os
import urllib.request

import cv2
import numpy as np
import pyfive

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # oculta avisos internos de OpenCV

# ---------------------------------------------------------------------------
# Operaciones de red neuronal (tensores H x W x C, sin dimension de lote)
# ---------------------------------------------------------------------------


def _relleno_same(x, k, s, d):
    """Relleno con ceros igual al padding='same' de TensorFlow."""
    k_efectivo = (k - 1) * d + 1

    def lados(n):
        salida = -(-n // s)
        total = max((salida - 1) * s + k_efectivo - n, 0)
        return total // 2, total - total // 2

    return np.pad(x, (lados(x.shape[0]), lados(x.shape[1]), (0, 0)))


def _rebanadas(x, k, s, d, padding):
    """Devuelve (ky, kx, rebanada) para cada posicion del kernel."""
    if padding == "same":
        x = _relleno_same(x, k, s, d)
    k_efectivo = (k - 1) * d + 1
    alto = (x.shape[0] - k_efectivo) // s + 1
    ancho = (x.shape[1] - k_efectivo) // s + 1
    for ky in range(k):
        for kx in range(k):
            y, xx = ky * d, kx * d
            yield ky, kx, x[y:y + s * (alto - 1) + 1:s, xx:xx + s * (ancho - 1) + 1:s]


def conv2d(x, kernel, s=1, padding="same", d=1):
    k, _, c_in, c_out = kernel.shape
    if k == 1 and s == 1:
        return (x.reshape(-1, c_in) @ kernel[0, 0]).reshape(x.shape[0], x.shape[1], c_out)
    salida = None
    for ky, kx, r in _rebanadas(x, k, s, d, padding):
        parcial = r.reshape(-1, c_in) @ kernel[ky, kx]
        salida = parcial if salida is None else salida + parcial
    return salida.reshape(r.shape[0], r.shape[1], c_out)


def depthwise_conv2d(x, kernel, s=1, padding="same", d=1):
    salida = None
    for ky, kx, r in _rebanadas(x, kernel.shape[0], s, d, padding):
        parcial = r * kernel[ky, kx, :, 0]
        salida = parcial if salida is None else salida + parcial
    return salida


def activacion(x, nombre):
    if nombre in (None, "linear"):
        return x
    if nombre == "relu":
        return np.maximum(x, 0)
    if nombre == "sigmoid":
        return 1 / (1 + np.exp(-x))
    if nombre == "softmax":
        e = np.exp(x - x.max())
        return e / e.sum()
    raise ValueError(f"Activacion no soportada: {nombre}")


def _leer_pesos_tfjs(manifiesto, carpeta):
    """Lee los pesos binarios de un modelo TF.js -> {nombre: array}."""
    pesos = {}
    for grupo in manifiesto:
        datos = b"".join(open(os.path.join(carpeta, p), "rb").read() for p in grupo["paths"])
        pos = 0
        for w in grupo["weights"]:
            if w.get("dtype", "float32") != "float32" or "quantization" in w:
                raise ValueError(f"Peso no soportado: {w['name']}")
            n = int(np.prod(w["shape"]))
            pesos[w["name"]] = np.frombuffer(datos, np.float32, n, pos).reshape(w["shape"])
            pos += n * 4
    return pesos


def _descargar_si_falta(url, ruta):
    if not os.path.exists(ruta):
        print(f"Descargando {os.path.basename(ruta)} (solo la primera vez)...")
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        urllib.request.urlretrieve(url, ruta + ".tmp")
        os.replace(ruta + ".tmp", ruta)
    return ruta


def recorte_cuadrado(img):
    """Recorta el centro de la imagen para dejarla cuadrada (como Teachable Machine)."""
    h, w = img.shape[:2]
    lado = min(h, w)
    y, x = (h - lado) // 2, (w - lado) // 2
    return img[y:y + lado, x:x + lado]


def recorte_cara(img, cara, margen=2.5):
    """Recorte cuadrado alrededor de una cara, con margen para incluir cabeza y cuello."""
    x, y, w, h = cara[:4]
    lado = int(max(w, h) * margen)
    x0 = max(int(x + w / 2) - lado // 2, 0)
    y0 = max(int(y + h / 2) - lado // 2, 0)
    return recorte_cuadrado(img[y0:y0 + lado, x0:x0 + lado])


# ---------------------------------------------------------------------------
# Modelo 1: imagen (Keras .h5)
# ---------------------------------------------------------------------------


class ClasificadorImagen:
    """Modelo de imagen de Teachable Machine exportado como Keras (.h5)."""

    def __init__(self, ruta_h5, ruta_etiquetas):
        archivo = pyfive.File(ruta_h5)
        try:
            config = archivo.attrs["model_config"]
            self.config = json.loads(config.decode() if isinstance(config, bytes) else config)
            self.pesos = {}
            archivo["model_weights"].visititems(self._guardar_peso)
        finally:
            archivo.close()

        with open(ruta_etiquetas, encoding="utf-8") as f:
            # Cada linea es "0 nombre": se quita el numero inicial
            self.etiquetas = [l.strip().split(" ", 1)[-1].strip() for l in f if l.strip()]
        self.tamano = self._buscar_tamano_entrada(self.config)

    def _guardar_peso(self, ruta, objeto):
        if hasattr(objeto, "shape"):  # es un dataset (los grupos no tienen forma)
            capa, nombre = ruta.split("/")[-2:]
            self.pesos.setdefault(capa, {})[nombre.split(":")[0]] = np.array(objeto[()], np.float32)

    def _buscar_tamano_entrada(self, capa):
        forma = capa.get("config", {}).get("batch_input_shape")
        if forma:
            return forma[1]
        capas = capa.get("config", {}).get("layers", [])
        return self._buscar_tamano_entrada(capas[0]) if capas else 224

    def predecir(self, imagen_rgb):
        """imagen_rgb: imagen cuadrada uint8. Devuelve las probabilidades de cada clase."""
        x = cv2.resize(imagen_rgb, (self.tamano, self.tamano), interpolation=cv2.INTER_AREA)
        x = x.astype(np.float32) / 127.5 - 1
        return self._ejecutar(self.config, x)

    def _ejecutar(self, capa, x):
        tipo, cfg = capa["class_name"], capa["config"]

        if tipo == "Sequential":
            for sub in cfg["layers"]:
                x = self._ejecutar(sub, x)
            return x

        if tipo in ("Functional", "Model"):
            valores = {}
            for sub in cfg["layers"]:
                if sub["class_name"] == "InputLayer":
                    valores[sub["name"]] = x
                    continue
                entradas = [valores[n[0]] for n in sub["inbound_nodes"][0]]
                valores[sub["name"]] = self._ejecutar(sub, entradas if len(entradas) > 1 else entradas[0])
            return valores[cfg["output_layers"][0][0]]

        p = self.pesos.get(cfg["name"], {})
        if tipo in ("InputLayer", "Dropout"):
            return x
        if tipo == "ZeroPadding2D":
            (t, b), (l, r) = cfg["padding"]
            return np.pad(x, ((t, b), (l, r), (0, 0)))
        if tipo == "Conv2D":
            x = conv2d(x, p["kernel"], cfg["strides"][0], cfg["padding"], cfg["dilation_rate"][0])
        elif tipo == "DepthwiseConv2D":
            x = depthwise_conv2d(x, p["depthwise_kernel"], cfg["strides"][0], cfg["padding"],
                                 cfg["dilation_rate"][0])
        elif tipo == "BatchNormalization":
            return (x - p["moving_mean"]) * (p["gamma"] / np.sqrt(p["moving_variance"] + cfg["epsilon"])) + p["beta"]
        elif tipo == "ReLU":
            return np.clip(x, 0, cfg.get("max_value") or np.inf)
        elif tipo == "Add":
            return sum(x)
        elif tipo == "GlobalAveragePooling2D":
            return x.mean(axis=(0, 1))
        elif tipo == "Dense":
            x = x @ p["kernel"]
        else:
            raise ValueError(f"Capa no soportada: {tipo}")

        if "bias" in p:
            x = x + p["bias"]
        return activacion(x, cfg.get("activation"))


# ---------------------------------------------------------------------------
# Modelo 2: pose (TF.js) = PoseNet + clasificador
# ---------------------------------------------------------------------------

URL_POSENET = "https://storage.googleapis.com/tfjs-models/savedmodel/posenet/mobilenet/float/{}/"

# Pares de puntos clave que se unen para dibujar el esqueleto
ESQUELETO = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
             (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


class PoseNet:
    """PoseNet MobileNetV1 (el mismo que usa Teachable Machine para extraer la pose)."""

    def __init__(self, carpeta, multiplicador=0.75, salto=16, resolucion=257):
        self.resolucion = resolucion
        self.salto = salto
        nombre_json = f"model-stride{salto}.json"
        ruta_json = os.path.join(carpeta, nombre_json)
        url = URL_POSENET.format({1.0: "100", 0.75: "075", 0.5: "050"}[multiplicador])

        with open(_descargar_si_falta(url + nombre_json, ruta_json), encoding="utf-8") as f:
            grafo = json.load(f)
        for grupo in grafo["weightsManifest"]:
            for archivo in grupo["paths"]:
                _descargar_si_falta(url + archivo, os.path.join(carpeta, archivo))

        self.constantes = _leer_pesos_tfjs(grafo["weightsManifest"], carpeta)
        self.nodos = [n for n in grafo["modelTopology"]["node"] if n["op"] != "Const"]

    def ejecutar(self, imagen_rgb):
        """Devuelve (heatmaps 17x17x17, offsets 17x17x34) para una imagen cuadrada."""
        x = cv2.resize(imagen_rgb, (self.resolucion, self.resolucion), interpolation=cv2.INTER_LINEAR)
        valores = dict(self.constantes)
        for nodo in self.nodos:
            op, attr = nodo["op"], nodo.get("attr", {})
            entradas = [valores[e] for e in nodo.get("input", [])]
            if op == "Placeholder":
                valores[nodo["name"]] = x.astype(np.float32) / 127.5 - 1
                continue
            if op in ("Conv2D", "DepthwiseConv2dNative"):
                s = int(attr["strides"]["list"]["i"][1])
                d = int(attr.get("dilations", {}).get("list", {}).get("i", [1, 1])[1])
                padding = base64.b64decode(attr["padding"]["s"]).decode().lower()
                funcion = conv2d if op == "Conv2D" else depthwise_conv2d
                resultado = funcion(entradas[0], entradas[1], s, padding, d)
            elif op == "BiasAdd":
                resultado = entradas[0] + entradas[1]
            elif op == "Relu6":
                resultado = np.clip(entradas[0], 0, 6)
            else:
                raise ValueError(f"Operacion no soportada: {op}")
            valores[nodo["name"]] = resultado

        heatmaps = activacion(valores["MobilenetV1/heatmap_2/BiasAdd"], "sigmoid")
        offsets = valores["MobilenetV1/offset_2/BiasAdd"]
        return heatmaps, offsets

    def puntos_clave(self, heatmaps, offsets, lado_imagen):
        """Posicion (x, y) y confianza de los 17 puntos del cuerpo."""
        alto, ancho, n = heatmaps.shape
        mejor = heatmaps.reshape(-1, n).argmax(axis=0)
        ys, xs = np.unravel_index(mejor, (alto, ancho))
        idx = np.arange(n)
        y = ys * self.salto + offsets[ys, xs, idx]
        x = xs * self.salto + offsets[ys, xs, idx + n]
        escala = lado_imagen / self.resolucion
        return np.stack([x, y], axis=1) * escala, heatmaps[ys, xs, idx]


class ClasificadorPose:
    """Modelo de pose de Teachable Machine (model.json + weights.bin + metadata.json)."""

    def __init__(self, carpeta):
        with open(os.path.join(carpeta, "metadata.json"), encoding="utf-8") as f:
            metadata = json.load(f)
        with open(os.path.join(carpeta, "model.json"), encoding="utf-8") as f:
            modelo = json.load(f)
        self.etiquetas = metadata["labels"]

        ajustes = metadata["modelSettings"]["posenet"]
        self.posenet = PoseNet(os.path.join(carpeta, "posenet"), ajustes["multiplier"],
                               ajustes["outputStride"], ajustes["inputResolution"])

        pesos = list(_leer_pesos_tfjs(modelo["weightsManifest"], carpeta).values())
        self.capas = []
        for capa in modelo["modelTopology"]["config"]["layers"]:
            cfg = capa["config"]
            if capa["class_name"] == "Dense":
                kernel = pesos.pop(0)
                bias = pesos.pop(0) if cfg["use_bias"] else 0
                self.capas.append((kernel, bias, cfg["activation"]))
            elif capa["class_name"] != "Dropout":
                raise ValueError(f"Capa no soportada: {capa['class_name']}")

    def predecir(self, imagen_rgb):
        """Devuelve (probabilidades, huella, puntos, confianzas) para una imagen cuadrada uint8.

        La huella es la salida de la capa oculta: sirve para comparar con un perfil registrado.
        """
        heatmaps, offsets = self.posenet.ejecutar(imagen_rgb)
        x = np.concatenate([heatmaps, offsets], axis=2).reshape(-1)
        huella = None
        for kernel, bias, act in self.capas:
            x = activacion(x @ kernel + bias, act)
            if huella is None:
                huella = x / (np.linalg.norm(x) or 1)
        puntos, confianzas = self.posenet.puntos_clave(heatmaps, offsets, imagen_rgb.shape[0])
        return x, huella, puntos, confianzas


# ---------------------------------------------------------------------------
# Modelo 3: voz (TF.js speech-commands)
# ---------------------------------------------------------------------------


class ClasificadorVoz:
    """Modelo de audio de Teachable Machine (model.json + weights.bin + metadata.json).

    Calcula el espectrograma igual que el navegador: cada SALTO muestras (a FRECUENCIA Hz) toma las
    ultimas 2 * SALTO, aplica la ventana de Blackman y guarda en decibelios las primeras frecuencias.
    El modelo mira los ultimos espectros (unos 43 = 1 segundo).
    """
    FRECUENCIA = 44100
    SALTO = 1024

    def __init__(self, carpeta):
        with open(os.path.join(carpeta, "metadata.json"), encoding="utf-8") as f:
            metadata = json.load(f)
        with open(os.path.join(carpeta, "model.json"), encoding="utf-8") as f:
            modelo = json.load(f)
        self.etiquetas = [e.strip() for e in metadata["wordLabels"]]
        self.pesos = _leer_pesos_tfjs(modelo["weightsManifest"], carpeta)

        capas = modelo["modelTopology"]["config"]["layers"]
        _, self.num_espectros, self.num_frecuencias, _ = capas[0]["config"]["batch_input_shape"]
        self.capas = []
        for capa in capas:  # el modelo es una cadena de capas; la ultima es un Sequential con la capa nueva
            if capa["class_name"] == "Sequential":
                self.capas += capa["config"]["layers"]
            elif capa["class_name"] not in ("InputLayer", "Dropout"):
                self.capas.append(capa)

        n = np.arange(2 * self.SALTO)
        self.ventana = 0.42 - 0.5 * np.cos(np.pi * n / self.SALTO) + 0.08 * np.cos(2 * np.pi * n / self.SALTO)

    def espectro(self, audio):
        """Una fila del espectrograma (en dB) a partir de las ultimas 2 * SALTO muestras."""
        x = audio[-2 * self.SALTO:] * self.ventana
        magnitud = np.abs(np.fft.rfft(x)[:self.num_frecuencias]) / len(x)
        return 20 * np.log10(np.maximum(magnitud, 1e-10))

    def predecir(self, espectrograma):
        """espectrograma: num_espectros x num_frecuencias (dB). Devuelve las probabilidades de cada clase."""
        x = (espectrograma - espectrograma.mean()) / (espectrograma.std() + 1e-7)  # como TF.js
        x = x[:, :, None].astype(np.float32)
        for capa in self.capas:
            tipo, cfg = capa["class_name"], capa["config"]
            if tipo == "Conv2D":
                x = self._convolucion(x, self.pesos[cfg["name"] + "/kernel"]) + self.pesos[cfg["name"] + "/bias"]
            elif tipo == "MaxPooling2D":
                x = self._max_pooling(x, cfg["pool_size"], cfg["strides"])
            elif tipo == "Flatten":
                x = x.reshape(-1)
            elif tipo == "Dense":
                x = x @ self.pesos[cfg["name"] + "/kernel"] + self.pesos[cfg["name"] + "/bias"]
            else:
                raise ValueError(f"Capa no soportada: {tipo}")
            x = activacion(x, cfg.get("activation"))
        return x

    @staticmethod
    def _convolucion(x, kernel):
        """Conv2D con padding='valid' y kernel rectangular."""
        kh, kw, _, c_out = kernel.shape
        alto, ancho = x.shape[0] - kh + 1, x.shape[1] - kw + 1
        salida = np.zeros((alto, ancho, c_out), np.float32)
        for ky in range(kh):
            for kx in range(kw):
                salida += x[ky:ky + alto, kx:kx + ancho] @ kernel[ky, kx]
        return salida

    @staticmethod
    def _max_pooling(x, tamano, salto):
        (ph, pw), (sh, sw) = tamano, salto
        alto, ancho = (x.shape[0] - ph) // sh + 1, (x.shape[1] - pw) // sw + 1
        return np.max([x[y:y + sh * (alto - 1) + 1:sh, xx:xx + sw * (ancho - 1) + 1:sw]
                       for y in range(ph) for xx in range(pw)], axis=0)


class Escucha:
    """Convierte el audio que va llegando en el espectrograma del ultimo segundo (y su volumen)."""

    def __init__(self, modelo, cada=11):
        self.modelo = modelo
        self.cada = cada  # se ejecuta el modelo cada 'cada' espectros nuevos (11 = ~0,25 s)
        self.audio = np.zeros(0, np.float32)  # muestras que aun no forman un espectro
        self.espectros = np.full((modelo.num_espectros, modelo.num_frecuencias), -200.0)
        self.niveles = np.full(modelo.num_espectros, -200.0)  # volumen de cada espectro (dBFS)
        self.historial = collections.deque(maxlen=200)  # (numero, espectro, volumen) de los ultimos ~4,6 s
        self.llenos = self.nuevos = 0  # llenos = numero del ultimo espectro calculado

    @property
    def nivel(self):
        """Volumen del momento mas fuerte del ultimo segundo, en dBFS (0 = maximo del microfono)."""
        return float(self.niveles.max())

    def agregar(self, muestras):
        """Suma audio nuevo. Devuelve True si ya toca ejecutar el modelo."""
        salto = self.modelo.SALTO
        self.audio = np.concatenate([self.audio, muestras])
        while len(self.audio) >= 2 * salto:
            self.espectros = np.roll(self.espectros, -1, axis=0)
            self.espectros[-1] = self.modelo.espectro(self.audio[:2 * salto])
            self.niveles = np.roll(self.niveles, -1)
            self.niveles[-1] = 10 * np.log10(np.mean(self.audio[salto:2 * salto] ** 2) + 1e-20)
            self.audio = self.audio[salto:]
            self.llenos, self.nuevos = self.llenos + 1, self.nuevos + 1
            self.historial.append((self.llenos, self.espectros[-1].copy(), self.niveles[-1]))
        if self.llenos >= self.modelo.num_espectros and self.nuevos >= self.cada:
            self.nuevos = 0
            return True
        return False


class DetectorFrase:
    """Recorta cada frase hablada (voz entre dos silencios) a partir de los espectros de Escucha.

    Hay voz cuando el volumen supera el minimo y el ruido de fondo (que se va midiendo) en 'margen' dB.
    """
    SILENCIO_FINAL = 17         # espectros sin voz que cierran la frase (~0,4 s)
    MINIMO, MAXIMO = 11, 130    # duracion valida de la frase en espectros (~0,25 s a ~3 s)

    def __init__(self, nivel_minimo, margen=12):
        self.nivel_minimo, self.margen = nivel_minimo, margen
        self.ruido = None    # volumen del ruido de fondo (dB)
        self.frase = None    # [(numero, espectro)] de la frase en curso
        self.anterior = None
        self.silencio = 0
        self.visto = 0       # ultimo espectro revisado
        self.aviso = ""

    @property
    def hablando(self):
        return self.frase is not None

    def actualizar(self, escucha):
        """Revisa los espectros nuevos. Devuelve (espectros, primero, ultimo) si acaba de terminar una frase."""
        terminada = None
        for numero, espectro, nivel in escucha.historial:
            if numero <= self.visto:
                continue
            self.visto = numero
            if self.ruido is None:
                self.ruido = nivel
            voz = nivel >= max(self.nivel_minimo, self.ruido + self.margen)
            if self.frase is None:
                if voz:
                    self.frase, self.silencio = [self.anterior or (numero, espectro), (numero, espectro)], 0
                else:
                    self.ruido = 0.95 * self.ruido + 0.05 * nivel
                self.anterior = (numero, espectro)
                continue
            self.frase.append((numero, espectro))
            self.silencio = 0 if voz else self.silencio + 1
            if len(self.frase) > self.MAXIMO:  # no para de sonar: es ruido o alguien hablando sin pausa
                self.aviso = "Frase demasiado larga (máximo 3 s): di solo tu palabra y calla."
                self.frase, self.ruido = None, nivel
            elif self.silencio >= self.SILENCIO_FINAL:
                frase = self.frase[:len(self.frase) - self.silencio + 2]
                self.frase = None
                if len(frase) >= self.MINIMO:
                    terminada = (np.array([e for _, e in frase]), frase[0][0], frase[-1][0])
                    self.aviso = ""
                else:
                    self.aviso = "Sonido demasiado corto."
        return terminada


# ---------------------------------------------------------------------------
# Caras: deteccion (YuNet) y huella facial (SFace), incluidos en OpenCV
# ---------------------------------------------------------------------------

URL_CARAS = "https://huggingface.co/opencv/{}/resolve/main/{}"
MODELOS_CARAS = [("face_detection_yunet", "face_detection_yunet_2023mar.onnx"),
                 ("face_recognition_sface", "face_recognition_sface_2021dec.onnx")]


class AnalizadorCaras:
    def __init__(self, carpeta):
        detector, reconocedor = [_descargar_si_falta(URL_CARAS.format(repo, archivo), os.path.join(carpeta, archivo))
                                 for repo, archivo in MODELOS_CARAS]
        self.detector = cv2.FaceDetectorYN.create(detector, "", (320, 320), 0.8)
        self.reconocedor = cv2.FaceRecognizerSF.create(reconocedor, "")

    def detectar(self, imagen_bgr):
        """Lista de caras: [x, y, ancho, alto, ojo1 (x, y), ojo2, nariz, boca1, boca2, confianza]."""
        self.detector.setInputSize((imagen_bgr.shape[1], imagen_bgr.shape[0]))
        _, caras = self.detector.detect(imagen_bgr)
        return [] if caras is None else list(caras)

    def recorte_alineado(self, imagen_bgr, cara):
        """Cara de 112x112 enderezada (ojos horizontales)."""
        return self.reconocedor.alignCrop(imagen_bgr, cara)

    def huella(self, cara_alineada):
        """128 numeros que describen la cara: dos fotos de la misma persona dan huellas parecidas."""
        f = self.reconocedor.feature(cara_alineada).reshape(-1)
        return f / np.linalg.norm(f)
