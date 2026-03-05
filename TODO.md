# TODO — Research Taste POC

Tracking de tareas pendientes, en progreso y bloqueadas. Actualizar cada vez que se completa o agrega algo.

---

## En progreso

- [ ] Obtener texto del paper Cerezer et al. 2023 para `data/papers/cerezer2023.txt`
- [ ] Configurar `.env` con credenciales Azure OpenAI reales

## Pendiente — Pipeline core

- [ ] Testear `generate_privi.py` con una tarea real
- [ ] Testear `generate_base.py` con una tarea real
- [ ] Testear `extract_forks.py` con par privi/base
- [ ] Testear `generate_interleaved.py`
- [ ] Testear `format_eval.py`

## Pendiente — Data

- [ ] Obtener texto de paper para sociology_bmi
- [ ] Obtener texto de paper para economics_immigration

## Pendiente — Evaluacion

- [ ] Disenar formato de evaluacion humana (UI o spreadsheet)
- [ ] Reclutar 3-5 evaluadores

## Completado

- [x] Crear estructura de directorios del proyecto
- [x] Escribir prompts del sistema (privi, base, step template)
- [x] Implementar `src/llm.py` (Azure OpenAI async client)
- [x] Implementar `src/common.py` (utilidades compartidas)
- [x] Implementar `src/extract.py` (extractor de tareas de DiscoveryBench)
- [x] Implementar `src/generate_privi.py` (generador trayectoria privilegiada)
- [x] Implementar `src/generate_base.py` (generador trayectoria base)
- [x] Implementar `src/generate_interleaved.py` (trayectoria intercalada)
- [x] Implementar `src/extract_forks.py` (extractor de forks/divergencias)
- [x] Implementar `src/format_eval.py` (formateador para evaluacion ciega)
- [x] Crear `.env.example` con variables Azure OpenAI
- [x] Crear `requirements.txt` (openai, pandas, python-dotenv)
- [x] Clonar DiscoveryBench en `data/discoverybench/`
- [x] Crear conda env `research-taste` (Python 3.11) e instalar deps
- [x] Migrar de anthropic SDK a Azure OpenAI (AsyncAzureOpenAI)
- [x] Crear sistema de documentacion (TODO.md, CHANGELOG.md, CLAUDE.md actualizado)
- [x] Adaptar `extract.py` a estructura real de DiscoveryBench (`real/train/<folder>/`)
- [x] Extraer las 3 tareas POC a `data/tasks/` (biology_fish, sociology_bmi, economics_immigration)
- [x] Verificar que archivos .dta (Stata) se leen con pandas — OK
- [x] Adaptar `common.py` para soportar .dta ademas de .csv
- [x] Testear `extract.py` con las 3 tareas — OK
- [x] Testear `build_dataset_summary` con las 3 tareas — OK
