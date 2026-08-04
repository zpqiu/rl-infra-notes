# Notas sobre infraestructura de RL

[Inglés](README_EN.md) | 中文

> Análisis en profundidad del código fuente de la infraestructura de entrenamiento de RL para LLM: programación asincrónica de RL, sincronización de pesos, precisión mixta FP8, precisión de enrutamiento MoE y más.

Notas de análisis en profundidad del código fuente de la infraestructura de entrenamiento de RL para LLM. No se trata solo de "qué es", sino de "por qué se diseñó así" y "cómo se implementa en el código".

## ¿Por qué este repositorio?

Cada vez hay más marcos de RL de código abierto, pero la mayoría de la documentación solo explica cómo usar la API. Sin embargo, cuando necesitas entender:

- ¿Cómo se programa exactamente la asignación de rollout y training en el entrenamiento RL asíncrono?
- ¿Qué ocurre en el motor de inferencia durante la sincronización de pesos? ¿abort o drain?
- ¿Qué operadores se cuantifican exactamente en el entrenamiento FP8? ¿Cuál es el formato de escala?
- ¿Qué problemas surgen con el topk del enrutador MoE en bf16?

Las respuestas solo están en el código fuente. Este repositorio registra de forma estructurada el proceso de "leer el código fuente", incluyendo la ubicación del código, tablas comparativas y diagramas de arquitectura.

## Notas

### [Entrenamiento RL asíncrono](docs/async-rl/)

Análisis comparativo de las decisiones de diseño de tres marcos en el entrenamiento RL asíncrono, que cubre las 4 dimensiones centrales del [Resumen de RL asíncrono de HuggingFace](https://huggingface.co/blog/async-rl-training-landscape): búfer de rollout, sincronización de pesos, gestión de obsolescencia y rollout parcial.

| Nota | Marco | Puntos destacados |
|------|--------|------------|
| [Guía de walkthrough de RL asíncrono de SLIME](docs/async-rl/slime-async-rl-walkthrough.md) | [THUDM/slime](https://github.com/THUDM/slime) | Programación de doble búfer, corrección de obsolescencia TIS + OPSM, mecanismo abort + recycle |
| [Guía de walkthrough de RL asíncrono de veRL](docs/async-rl/verl-async-rl-walkthrough.md) | [volcengine/verl](https://github.com/volcengine/verl) | Cola acotada + presión inversa, broadcast acotado de NCCL, MIS multi-versión IS, continuación de prefijo |
| [Guía de walkthrough de RL asíncrono de NeMo-RL](docs/async-rl/nemo-rl-async-rl-walkthrough.md) | [NVIDIA/NeMo-RL](https://github.com/NVIDIA/NeMo-RL) | Búfer de reutilización + coincidencia de pesos objetivo, actualización de pesos en vuelo, TIS / ICE-POP / seq-mask-TIS |

### Entrenamiento e inferencia con precisión mixta FP8

Análisis detallado de los detalles de la cuantificación en FP8, el formato de escala y la precisión de comunicación en el entrenamiento e inferencia.

| Nota | Marco | Puntos destacados |
|------|--------|------------|
| [Visión general de Megatron](docs/fp8/megatron-overview.md) | Megatron-LM / Bridge / TE | Relaciones entre componentes, rango de cuantificación blockwise FP8 |
| [Explicación detallada de fp8_param_gather](docs/fp8/fp8-param-gather.md) | Megatron-LM | Optimización de comunicación all-gather en FP8, comparación de procesos de actualización de parámetros |
| [Análisis de escala blockwise FP8](docs/fp8/fp8_blockwise_scale_analysis.md) | vLLM | DeepGEMM UE8M0 vs escala FP32, prioridad de despacho de kernels |
| [Análisis del tipo de dato del enrutador MoE](docs/fp8/megatron_moe_router_dtype_analysis.md) | Megatron-LM + vLLM | Seguimiento del tipo de dato de todo el enrutador MoE (entrenamiento vs inferencia), riesgo de precisión topk en bf16 |
| [Análisis de no determinismo en el unpermute de MoE](docs/fp8/moe_unpermute_determinism.md) | Megatron-LM + TE | Causa principal de la no determinismo de scatter_add_, kernel de Triton gather-reduce de TE, construcción de row_id_map en 3 pasos |

### 🌐 Traducciones al inglés

Todas las notas tienen una versión en inglés ubicada en el directorio [`docs-en/`](docs-en/), con una estructura completamente paralela a `docs/`.

## Marcos estudiados

| Marco | Enfoque |
|--------|---------|
| [NVIDIA NeMo-RL](https://github.com/NVIDIA/NeMo-RL) | Pipeline de entrenamiento RL, GRPO asíncrono |
| [veRL](https://github.com/volcengine/verl) | RL asíncrono, sincronización de pesos |
| [SLIME](https://github.com/THUDM/slime) | RL asíncrono, TIS/OPSM |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | Entrenamiento distribuido, FP8, MoE |
| [Megatron-Bridge](https://github.com/NVIDIA/Megatron-Bridge) | Conversión HF↔Megatron |
| [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) | Kernels FP8 |
| [vLLM](https://github.com/vllm-project/vllm) | Inferencia, FP8, enrutamiento MoE |

## Contribución

¡Bienvenidos a crear problemas para discutir o agregar análisis! Si encuentras que las referencias al código en las notas están desactualizadas (los marcos cambian rápidamente), también son bienvenidos los PR para corregirlas.

## Licencia

[MIT](LICENSE)
