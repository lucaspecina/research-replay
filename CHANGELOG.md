# CHANGELOG — Research Taste POC

Registro cronologico de cambios significativos. Formato: fecha, descripcion, archivos afectados.

---

## 2026-03-18 — Arquitectura SREG-aligned + privi-anchored pairs

### Problema identificado: pares DPO invalidos
- Las trayectorias independientes (privi y base) divergen despues del step 1
- Comparar step N de una con step N de la otra no es un par valido: vienen de estados distintos
- Para DPO necesitamos pares desde el MISMO estado

### Nueva arquitectura alineada con synthetic-research-envs (SREG)
- `src/python_exec.py` — Persistent namespace (reemplaza subprocess). Variables viven entre steps como un Jupyter notebook
- `src/trajectory_runner.py` — TrajectoryRunner (como EpisodeRunner de SREG). Gestiona estado, steps, code execution, snapshots
- `src/verifier.py` — VerifierTool (single authority scoring). Compara decisiones contra gold hypothesis
- `src/generate_anchored.py` — Two-pass privi-anchored: (1) corre privi con code real, (2) en cada checkpoint pregunta al base que haria desde el MISMO estado
- `src/format_dpo.py` — Export a JSONL compatible con TRL
- `src/training/rubric.py` — Reward dispatcher (como SREG)
- `src/training/env.py` — Stub para verifiers.StatefulToolEnv

### Primer anchored run (biology_fish)
- 6 pares limpios desde estados identicos (vs 4 aproximados del pipeline viejo)
- Divergencias: 0.44 a 1.00 — todos usables
- Persistent namespace validado: variables de step 1 accesibles en step 4

### Papers y tareas nuevas
- 4 papers open access encontrados y guardados: Appiah 2017, Brozio 2024, Heyard 2024, Riera 2024
- 4 tareas nuevas extraidas de DiscoveryBench test split
- `extract.py` extendido para soportar train + test splits
- Pipeline agentic independiente corrido en 5 tareas (20 fork pairs total)

---

## 2026-03-18 — Semi-agentic pipeline + primer end-to-end

### Cambio de paradigma: simulated -> semi-agentic
- Las trayectorias ahora ejecutan codigo real sobre datos reales (antes el modelo inventaba los resultados)
- Loop por paso: LLM propone analisis + escribe Python -> subprocess ejecuta en CSV real -> resultado real se devuelve al LLM
- Ambos generadores (`generate_privi.py`, `generate_base.py`) soportan `--mode agentic` y `--mode simulated`

### Archivos nuevos
- `src/sandbox.py` — Ejecucion de codigo via subprocess con timeout, preamble auto-generado que carga DataFrames, sanitizacion de newlines escapados
- `src/generate_loop.py` — Loop compartido semi-agentic usado por privi y base (multi-turn conversation con resultados reales)
- `prompts/step_agentic.txt` — Template para pasos agentic (pide reasoning + action + code ejecutable)

### Archivos modificados
- `src/llm.py` — Agregada `call_messages()` para conversaciones multi-turn (message array)
- `src/common.py` — Agregada `build_df_description()` para describir DataFrames disponibles en el prompt
- `src/generate_privi.py` — Refactorizado: `--mode agentic` (default) usa generate_loop, `--mode simulated` mantiene comportamiento original
- `src/generate_base.py` — Idem

### Primer end-to-end exitoso (biology_fish)
- Privi agentic: 6 pasos con codigo real, R²=0.925, MBL_evol coef mas alto
- Base agentic: 6 pasos, incluyo variables circulares (R²=1.0), uso RF (inestable), conclusion ambigua
- 4 forks extraidos con divergencias reales en juicio cientifico
- Pares ciegos generados en `eval/biology_fish_pairs.json`

### Hallazgos
- gpt-5.4 escapa newlines como `\n` literal en campo `code` del JSON en ~50% de primeros pasos. Fix: `sanitize_code()` en sandbox.py
- El mecanismo semi-agentic produce divergencias mucho mas ricas que el simulado: el privi elige hierarchical partitioning, el base usa RF feature importance
- El base cometio error real (incluir BAMM_NetDiv como predictor → R²=1.0 circular) que nunca hubiera aparecido en modo simulado

### Dependencias nuevas
- scipy, scikit-learn, statsmodels (para que el codigo generado pueda hacer analisis estadisticos reales)

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

---

## 2026-03-05 — Migracion a Azure AI Foundry v1 API

### LLM client reescrito
- `src/llm.py`: `AsyncAzureOpenAI` -> `AsyncOpenAI` con `base_url` (patron v1 recomendado por Microsoft)
- Eliminada dependencia de `api_version` — v1 API no la necesita
- Env vars renombradas: `AZURE_API_BASE` -> `AZURE_OPENAI_BASE_URL`, `AZURE_API_KEY` -> `AZURE_INFERENCE_CREDENTIAL`
- Ahora compatible con cualquier modelo del catalogo Foundry (GPT, DeepSeek, Llama, etc.)

### Archivos afectados
- `src/llm.py` — cliente migrado a v1 API
- `.env.example` — nuevas variables de entorno
- `CLAUDE.md` — actualizada seccion de convenciones Azure
- `README.md` — actualizado setup con nuevas env vars
