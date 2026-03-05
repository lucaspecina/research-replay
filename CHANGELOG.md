# CHANGELOG — Research Taste POC

Registro cronologico de cambios significativos. Formato: fecha, descripcion, archivos afectados.

---

## 2026-03-05 — Setup inicial del proyecto

### Estructura y scaffolding
- Creada estructura completa de directorios: `src/`, `prompts/`, `data/`, `trajectories/`, `eval/`, `results/`
- Creados `.gitkeep` para preservar carpetas vacias en git
- Creado `.env.example` con variables para Azure OpenAI
- Creado `requirements.txt`: openai, pandas, python-dotenv
- Actualizado `.gitignore` con entradas del proyecto (data/discoverybench, trajectories JSON, etc.)

### Prompts
- `prompts/privi_system.txt` — System prompt para modelo privilegiado (tiene paper)
- `prompts/base_system.txt` — System prompt para modelo base (sin paper)
- `prompts/step_template.txt` — Template para generacion paso a paso

### Scripts del pipeline
- `src/llm.py` — Cliente Azure OpenAI compartido (async, semaphore rate limiting, retry con backoff, timeout, JSON extraction)
- `src/common.py` — Utilidades compartidas (load_task, load_paper, build_dataset_summary, save_json, format_steps)
- `src/extract.py` — Extractor de tareas desde metadata de DiscoveryBench
- `src/generate_privi.py` — Generador de trayectorias privilegiadas (con paper)
- `src/generate_base.py` — Generador de trayectorias base (sin paper)
- `src/generate_interleaved.py` — Generador de trayectorias intercaladas (base/privi alternan pasos)
- `src/extract_forks.py` — Extractor de divergencias entre privi y base via LLM
- `src/format_eval.py` — Formateador de pares para evaluacion humana ciega (randomiza orden)

### Infraestructura
- Clonado DiscoveryBench en `data/discoverybench/` (estructura real: `real/train/<folder>`)
- Creado conda env `research-taste` (Python 3.11.14)
- Instaladas dependencias: openai 2.24.0, pandas 3.0.1, python-dotenv 1.2.2

### Migracion anthropic -> Azure OpenAI
- Todos los scripts migrados de `anthropic` SDK a `openai` SDK (AsyncAzureOpenAI)
- Centralizado cliente LLM en `src/llm.py` con patron async + retry + rate limiting
- Extraida logica compartida a `src/common.py` para eliminar duplicacion

### Documentacion
- `CLAUDE.md` — Actualizado con tech stack Azure OpenAI, estructura con llm.py y common.py
- `PROJECT.md` — Descripcion completa del proyecto y motivacion (sin cambios)
- `TODO.md` — Tracking de tareas (creado)
- `CHANGELOG.md` — Este archivo (creado)

### Hallazgos
- DiscoveryBench estructura real: `discoverybench/real/train/<folder_name>/` (no `real/<domain>/`)
- Carpetas: `evolution_freshwater_fish`, `nls_bmi`, `nls_bmi_raw`, `immigration_offshoring_effect_on_employment`
- Cada carpeta tiene `metadata_N.json` + archivos de datos (CSV o DTA)

---

## 2026-03-05 — Adaptacion a datos reales de DiscoveryBench

### extract.py reescrito
- Ahora apunta a `real/train/<folder_name>/` en vez de `real/<domain>/`
- TASK_ALIASES mapean a folders reales: `evolution_freshwater_fish`, `nls_bmi`, `immigration_offshoring_effect_on_employment`
- Parsea formato real de columnas: `{"raw": [{"name": ..., "description": ...}]}` -> dict plano `{name: desc}`
- Parsea queries nested: `[[{qid, question, true_hypothesis}]]` -> lista flat
- Agrega flag `--list` para listar todas las tareas disponibles
- Agrega `--metadata-index` para elegir variante de metadata

### common.py actualizado
- `build_dataset_summary` ahora soporta archivos `.dta` (Stata) via `pd.read_stata()`

### Tareas extraidas
- `data/tasks/biology_fish.json` — biology, 1 dataset (CSV, 460x21), 3 queries
- `data/tasks/sociology_bmi.json` — sociology, 1 dataset (CSV, 12686x9), 2 queries
- `data/tasks/economics_immigration.json` — economics, 2 datasets (DTA, 464x3 + 464x4), 2 queries

### Hallazgos
- Formato de metadata de DiscoveryBench: columns son `{"raw": [{name, description, depth}]}`, no dict plano
- Queries son listas de listas (agrupadas), no lista flat
- Los 3 archivos .dta se leen sin problemas con `pd.read_stata()`
- Biology tiene 4 metadata variants (0-3), sociology 6 (0-5), economics 2 (0-1)
- Usamos metadata_0 para todas las tareas POC (query mas representativa)

---

## 2026-03-05 — Paper stub y verificacion de imports

### Paper
- Creado `data/papers/cerezer2023.txt` como stub (reconstruido desde metadata + PROJECT.md)
- Incluye: abstract, methods, key results, discussion, workflow
- Nature bloquea web fetch directo — pendiente reemplazar con paper real

### Verificacion
- Todos los imports de `src/llm.py` y `src/common.py` funcionan correctamente
- `extract_json` parsea JSON directo y en code blocks
- Pipeline listo para testear con credenciales Azure OpenAI reales
