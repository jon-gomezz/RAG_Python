# Decisiones técnicas

Este documento explica **en profundidad** las decisiones de diseño del proyecto:
qué se eligió, por qué, qué alternativas se descartaron y qué contrapartidas tiene
cada decisión. El objetivo es que el código sea comprensible y justificable, no solo
funcional.

## Índice

1. [Contexto y principios](#1-contexto-y-principios)
2. [Framework: FastAPI](#2-framework-fastapi)
3. [Arquitectura por capas e inyección de dependencias](#3-arquitectura-por-capas-e-inyección-de-dependencias)
4. [Carga y normalización de documentos](#4-carga-y-normalización-de-documentos)
5. [Fragmentación (chunking)](#5-fragmentación-chunking)
6. [Recuperación: TF-IDF y modo semántico](#6-recuperación-tf-idf-y-modo-semántico)
7. [Generación de respuestas fundamentadas](#7-generación-de-respuestas-fundamentadas)
8. [Configuración y secretos](#8-configuración-y-secretos)
9. [Manejo de errores](#9-manejo-de-errores)
10. [Almacenamiento en memoria](#10-almacenamiento-en-memoria)
11. [Estrategia de tests](#11-estrategia-de-tests)
12. [Calidad de código e integración continua](#12-calidad-de-código-e-integración-continua)
13. [Limitaciones y trabajo futuro](#13-limitaciones-y-trabajo-futuro)

---

## 1. Contexto y principios

El reto pide una solución que cargue documentos (`.txt`, `.md`, `.pdf`) y responda
preguntas **usando solo su contenido**, mostrando las fuentes y avisando cuando no
haya información suficiente.

Principios que guiaron todas las decisiones:

- **Trazabilidad**: cada respuesta debe poder justificarse con los fragmentos exactos
  que la sustentan. Nunca se ocultan las fuentes.
- **Honestidad**: si no hay contexto suficiente, se dice claramente en lugar de
  inventar (evitar "alucinaciones" del modelo).
- **Simplicidad y determinismo por defecto**: preferir lo simple y predecible
  (fácil de testear y de explicar) antes que lo sofisticado, salvo que aporte un
  valor claro.
- **Separación de responsabilidades**: cada pieza hace una cosa y es sustituible.

---

## 2. Framework: FastAPI

**Decisión:** exponer la solución como una API REST con **FastAPI**.

**Por qué:**

- **Validación y tipado integrados** (vía Pydantic): los contratos de entrada/salida
  se declaran como modelos y se validan solos.
- **Documentación interactiva automática** (`/docs`, Swagger UI): permite probar el
  servicio sin escribir código, muy útil para una evaluación.
- **Inyección de dependencias nativa**, que encaja con el diseño desacoplado y la
  testeabilidad (ver §3 y §11).
- **Asíncrono y ligero**, suficiente para el alcance.

**Alternativas descartadas:**

- **Flask**: válido y simple, pero requiere más trabajo manual para validación,
  documentación y tipado.
- **Django**: demasiado pesado (ORM, admin, etc.) para un servicio sin base de datos.
- **CLI**: el reto lo permitía, pero una API es más demostrable y reutilizable.

---

## 3. Arquitectura por capas e inyección de dependencias

**Decisión:** separar el código en capas con responsabilidades claras:

- `app/main.py` — endpoints HTTP **delgados** (solo validan/serializan y delegan).
- `app/services/` — la lógica de negocio (carga, troceado, recuperación, generación,
  orquestación).
- `app/store/` — almacenamiento.
- `app/core/` — configuración, excepciones y manejadores de error transversales.

**Por qué:**

- La lógica de negocio **no depende de FastAPI**: se podría reutilizar desde una CLI,
  un script o tests sin tocar nada.
- Los endpoints delgados son fáciles de leer y mantener.
- La orquestación vive en una sola fachada (`RAGPipeline`), que coordina las piezas.

**Inyección de dependencias:** las piezas que hablan con el exterior (cliente LLM,
cliente de embeddings, recuperador) se **inyectan** en lugar de instanciarse dentro.
Esto permite:

- **Testear sin servicios externos** (se inyectan dobles de prueba).
- **Intercambiar implementaciones** (p.ej. TF-IDF ↔ embeddings) sin tocar la lógica.

El cliente de OpenAI se crea de forma **perezosa** (`LazyOpenAILLMClient`): la
aplicación arranca aunque no haya `LLM_API_KEY`, y la clave solo se exige al generar
una respuesta de verdad (no al subir documentos ni en `/health`).

---

## 4. Carga y normalización de documentos

**Decisión:** un único punto de entrada `load_document(filename, content)` que valida
la extensión y delega en un loader por formato (`.txt`/`.md` por decodificación UTF-8;
`.pdf` con `pypdf`). El texto extraído se **normaliza** (espacios, tabulaciones y
saltos de línea sobrantes).

**Por qué:**

- La extracción de PDFs suele producir texto "sucio"; normalizar mejora la calidad de
  los fragmentos y de las fuentes mostradas.
- La decodificación UTF-8 es **tolerante** (`errors="replace"`): un byte corrupto
  puntual no tira todo el documento.
- Validar la extensión por nombre es simple y suficiente para el alcance.

**Contrapartida:** no se inspecciona el contenido real del fichero (solo la extensión),
y `pypdf` puede no extraer bien PDFs escaneados (imágenes sin texto).

---

## 5. Fragmentación (chunking)

**Decisión:** trocear el texto por **número de caracteres** con **solapamiento**
configurable, guardando metadatos por fragmento (origen, índice, posición), y
**deduplicar** fragmentos idénticos antes de indexar.

**Por qué:**

- **Por caracteres** es determinista y trivial de testear (frente a trocear por
  tokens, que depende del modelo).
- El **solapamiento** reduce el riesgo de partir una idea entre dos fragmentos.
- Los **metadatos** (posición, origen) son la base de la trazabilidad.
- El **deduplicado** evita que contenido repetido (cabeceras, índices que se repiten
  entre páginas) ocupe varias posiciones en la recuperación.

**Alternativas / mejoras futuras:** troceado por tokens (más alineado con el LLM) o
troceado **consciente de la estructura** (respetar párrafos, tablas, secciones), que
evitaría partir tablas a la mitad.

---

## 6. Recuperación: TF-IDF y modo semántico

**Decisión:** recuperación con **TF-IDF + similitud del coseno** por defecto, detrás de
una **interfaz común `Retriever`**, con un **modo semántico opcional con embeddings**
seleccionable por configuración (`RETRIEVAL_MODE`).

**Por qué TF-IDF por defecto:**

- **Determinista**: la misma pregunta da siempre el mismo resultado → fácil de testear
  y de explicar.
- **Sin dependencias externas ni coste**: no necesita internet ni clave para buscar.
- **Suficiente** para el alcance del reto.

**Por qué añadir embeddings (opcional):** TF-IDF compara **palabras literales**, así
que falla con sinónimos o con preguntas en un idioma distinto al del documento. Los
embeddings comparan **significado**, resolviendo esos casos.

**Cómo se hizo intercambiable:** ambos recuperadores implementan la misma interfaz
`Retriever` (`index` / `retrieve`). El `RAGPipeline` depende de la interfaz, no de una
implementación concreta, así que cambiar de modo no toca la lógica.

| | TF-IDF (por defecto) | Embeddings (opcional) |
| --- | --- | --- |
| Tipo de comparación | Léxica (palabras) | Semántica (significado) |
| Sinónimos / idiomas | No | Sí |
| Coste | Gratis | Llamadas de pago |
| Latencia | Instantánea | Mayor |
| Determinista | Sí | Menos |
| Requiere clave | No | Sí |

---

## 7. Generación de respuestas fundamentadas

**Decisión:** construir un prompt **fundamentado** que obliga al modelo a responder
**solo** con el contexto recuperado, y **detectar la falta de contexto antes de llamar
al LLM**.

**Por qué:**

- El prompt incluye los fragmentos numerados con su fuente e instruye explícitamente
  no usar conocimiento externo y avisar si no hay respuesta en el contexto. Esto
  reduce las "alucinaciones".
- Si la recuperación no devuelve fragmentos, se responde con el mensaje de
  "información insuficiente" **sin llamar al LLM**: ahorra coste y latencia y evita que
  el modelo improvise sin material.

**Cliente LLM inyectable:** la generación depende de una interfaz `LLMClient`, no de
OpenAI directamente. Permite cambiar de proveedor y, sobre todo, **testear con un doble**
sin llamadas reales. Los fallos inesperados del proveedor se reempaquetan en un error
de dominio (`LLMGenerationError`) para no propagar errores técnicos crudos.

---

## 8. Configuración y secretos

**Decisión:** toda la configuración se lee de **variables de entorno** mediante
`pydantic-settings`, centralizada en `Settings`, con valores por defecto y validación.

**Por qué:**

- **Un único lugar** para la configuración (frente a esparcir `os.environ`).
- **Validación de tipos y rangos** (p.ej. `min_relevance_score` entre 0 y 1;
  `retrieval_mode` solo acepta `tfidf`/`embeddings`).
- **Los secretos no van en el código**: la clave se lee de `.env` (ignorado por Git);
  el repositorio solo incluye `.env.example` con los nombres de las variables, vacías.

Esto cumple el requisito de "configuración por variables de entorno" y mantiene la
clave fuera del control de versiones.

---

## 9. Manejo de errores

**Decisión:** definir **excepciones de dominio** (todas heredan de `RAGError`) y
**traducirlas a HTTP en un único lugar** (`register_exception_handlers`), con un cuerpo
de respuesta uniforme `{"detail": ...}`.

**Por qué:**

- Las excepciones describen el problema en términos del negocio (extensión no
  soportada, fichero vacío, falta de clave...), **independientes de HTTP**. Esto las
  hace reutilizables fuera de una API.
- La traducción a códigos HTTP vive en la capa de API, en un solo sitio, lo que
  mantiene los endpoints delgados y los mensajes coherentes.

Mapa de traducción:

| Excepción | HTTP | Significado |
| --- | --- | --- |
| `UnsupportedFileTypeError` | 400 | Petición inválida del cliente |
| `EmptyFileError` / `DocumentLoadError` | 422 | Entrada no procesable |
| `MissingAPIKeyError` | 503 | Servicio mal configurado |
| `LLMGenerationError` / `EmbeddingError` | 502 | Fallo del proveedor externo |
| `RAGError` (fallback) | 500 | Error inesperado de dominio |

---

## 10. Almacenamiento en memoria

**Decisión:** guardar los fragmentos en un **almacén en memoria** (`DocumentStore`),
sin persistencia en disco ni base de datos.

**Por qué:**

- El reto pedía una solución **sencilla**; en memoria es suficiente para demostrar el
  flujo completo y simplifica el despliegue (sin dependencias de BD).
- Mantiene el foco en la parte RAG, no en la infraestructura.

**Contrapartida (limitación conocida):** los documentos **se pierden al reiniciar** el
servicio. La mejora natural sería persistir el índice en disco o en una base de datos.

---

## 11. Estrategia de tests

**Decisión:** tests unitarios y de API que **no dependen de servicios externos**,
usando dobles de prueba e inyección de dependencias.

**Cómo:**

- **Dobles del LLM y del embedder**: implementan la interfaz y devuelven valores fijos,
  permitiendo probar toda la cadena **sin llamar a OpenAI ni gastar**.
- **API**: se prueba con el `TestClient` de FastAPI, sustituyendo la tubería con
  `app.dependency_overrides` por una con LLM simulado.
- **Evaluación de recuperación** (`test_retrieval_eval.py`): un pequeño corpus
  etiquetado verifica que cada pregunta recupera el fragmento esperado (precisión
  top-1), como red de seguridad ante regresiones.
- **Casos límite**: ficheros vacíos, extensión no soportada, preguntar sin documentos,
  falta de clave, fallo del proveedor.

**Por qué:** tests deterministas, rápidos, gratis y reproducibles en CI.

---

## 12. Calidad de código e integración continua

**Decisión:** automatizar calidad y verificación:

- **`ruff`** (linter + formateo), **`mypy`** (tipos) y **`pre-commit`** (revisión antes
  de cada commit).
- **Versiones fijadas** en `requirements.txt` (runtime) y `requirements-dev.txt` (dev)
  para reproducibilidad.
- **Integración continua** (GitHub Actions): en cada push y pull request se ejecutan
  linter, formato, tipos y tests.

**Por qué:** garantiza un estilo consistente, detecta errores de tipos antes de
ejecutar, hace el entorno reproducible y asegura que `main` siempre está sano.

---

## 13. Limitaciones y trabajo futuro

- **TF-IDF es léxico**: mitigado con el modo `embeddings`, pero el modo por defecto no
  entiende sinónimos ni cruza idiomas.
- **Chunking por caracteres**: puede partir tablas o estructuras; un troceado
  consciente de la estructura mejoraría la calidad.
- **Páginas tipo índice**: el deduplicado elimina fragmentos idénticos, pero un
  re-ranking más fino reduciría aún más el ruido.
- **Almacenamiento en memoria**: sin persistencia; los documentos se pierden al
  reiniciar.
- **Observabilidad**: el logging es básico; en producción convendría logging
  estructurado y métricas.
