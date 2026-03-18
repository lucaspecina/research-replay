# TODO — Research Taste POC

Tracking de tareas pendientes, en progreso y bloqueadas. Actualizar cada vez que se completa o agrega algo.

---

## En progreso

- [ ] Correr `generate_anchored.py` en las 4 tareas restantes (archaeology, meta_regression, plants_pathways, worldbank_education)
- [ ] Reemplazar paper stub de Cerezer 2023 con texto completo del paper real

## Pendiente — Data

- [ ] Obtener texto de paper para sociology_bmi y economics_immigration (tareas originales train)
- [ ] Escalar anchored a todas las variantes de metadata de DiscoveryBench (~60-80 combinaciones)

## Pendiente — Pipeline

- [ ] Implementar base-anchored mode (DAgger) en `generate_anchored.py --anchor base`
- [ ] Refactorizar `generate_loop.py` para usar `TrajectoryRunner`
- [ ] Multiples runs por tarea para medir varianza

## Pendiente — Training

- [ ] Testear `format_dpo.py` con pares anchored reales
- [ ] Elegir modelo target para fine-tuning (Llama-3.1-8B? DeepSeek-R1-Distill?)
- [ ] Implementar `training/env.py` completo con verifiers library
- [ ] Setup Azure ML compute para DPO training

## Pendiente — Evaluacion

- [ ] Disenar formato de evaluacion humana (UI o spreadsheet)
- [ ] Reclutar 3-5 evaluadores

## Completado

- [x] Implementar arquitectura SREG-aligned (python_exec, trajectory_runner, verifier)
- [x] Crear `src/python_exec.py` (persistent namespace, reemplaza sandbox subprocess)
- [x] Crear `src/trajectory_runner.py` (estado del episodio con snapshots)
- [x] Crear `src/verifier.py` (scoring centralizado)
- [x] Crear `src/generate_anchored.py` (pares DPO privi-anchored, two-pass)
- [x] Crear `src/format_dpo.py` (export JSONL para TRL)
- [x] Crear `src/training/rubric.py` (reward dispatcher)
- [x] Crear `src/training/env.py` (stub verifiers.StatefulToolEnv)
- [x] Validar anchored mode en biology_fish — 6 pares limpios desde estados identicos
- [x] Buscar y obtener 4 papers open access (Appiah 2017, Brozio 2024, Heyard 2024, Riera 2024)
- [x] Extraer 4 tareas nuevas de DiscoveryBench test split
- [x] Extender `extract.py` para soportar train + test splits
- [x] Correr pipeline agentic independiente en 5 tareas (20 fork pairs total)
- [x] Configurar `.env` con credenciales Azure AI Foundry
- [x] Implementar modo semi-agentic (code execution en datos reales)
- [x] Crear `src/sandbox.py` (ejecucion segura de codigo via subprocess)
- [x] Crear `src/generate_loop.py` (loop compartido privi/base)
- [x] Agregar `call_messages()` a `src/llm.py` (multi-turn)
- [x] Agregar `build_df_description()` a `src/common.py`
- [x] Crear `prompts/step_agentic.txt`
- [x] Testear pipeline end-to-end agentic con biology_fish (privi + base + forks + eval)
- [x] Generar trayectoria privi agentic biology_fish — divergencias reales encontradas
- [x] Generar trayectoria base agentic biology_fish — comportamiento generico confirmado
- [x] Extraer 4 fork pairs con divergencias significativas en juicio cientifico
- [x] Formatear pares ciegos para evaluacion humana
- [x] Fix bug de newlines escapados en codigo generado por LLM
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
