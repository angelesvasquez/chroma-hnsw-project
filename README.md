
# HNSW Modificado: Skip Connections + ADSampling/PCA

Versión modificada de la implementación de HNSW (Hierarchical Navigable Small World) de hnswlib. Basado en: https://github.com/chroma-core/hnswlib

## Objetivo

Mejorar HNSW mediante dos modificaciones:

-   **Skip Connections**: agrega conexiones adicionales entre nodos cercanos en el espacio vectorial pero no conectados directamente en el grafo. Reduce mínimos locales durante la búsqueda greedy.
-   **ADSampling + PCA Pruning**: proyecta los vectores con PCA y usa un umbral epsilon para descartar candidatos sin calcular la distancia completa. Reduce el costo de exploración.

## Estructura del proyecto

```
chroma-hnsw-project/
├── hnswlib/
│   └── hnswlib/
│       ├── hnswalg.h        # implementación del algoritmo (Skip + ADSampling)
│       └── hnswlib.h
├── benchmark/
│   ├── benchmark.cpp        # evaluacion de rendimiento
│   └── siftsmall/           # dataset pequeño
└── README.md

```

## Datasets

**SIFT Small** (incluido, usado por defecto): `benchmark/siftsmall/`, 10,000 vectores base, 100 queries, dim=128.

**SIFT 1M** (descarga externa, para pruebas de escalabilidad): `ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz`, 1,000,000 vectores base.

## Métricas evaluadas

-   Recall@10
-   QPS
-   Build time (segundos)
-   Prune rate (solo ADSampling)

## Parámetros principales

-   M = 16
-   ef_construction = 200
-   ef_search ∈ {10, 20, 50, 100}
-   epsilon (ADSampling): variable experimental

## Compilación

```
g++ -O3 -o benchmark benchmark.cpp

```

## Ejecución

```
./benchmark [base_path] [query_path] [gt_path]

```

Por defecto usa los archivos en `siftsmall/`. El programa corre automáticamente, en un solo binario:

1.  HNSW original
2.  Skip Connections + ADSampling (combinado)
3.  Solo Skip Connections
4.  Solo ADSampling/PCA
5.  Barrido de epsilon (ef_search=50 y ef_search=100)

Cada configuración construye el índice una sola vez y evalúa todos los `ef_search` sobre ese mismo grafo.

## Integración con Chroma

Para integrar esta versión modificada de HNSW con Chroma se debe realizar la sustitución de la dependencia original de `hnswlib`, la cual actualmente apunta al repositorio oficial en GitHub.

----------

### 1. Clonar Chroma

```
git clone https://github.com/chroma-core/chroma.git
```

----------

### 2. Ubicación del repositorio modificado

Colocar este proyecto (`hnswlib/` con `hnswalg.h` modificado) en un directorio paralelo al repositorio de Chroma:

```
workspace/
├── chroma/
└── hnswlib/
```

Esto permite referenciar la implementación local mediante ruta relativa.

----------

### 3. Modificación de la dependencia

En el archivo `Cargo.toml` de Chroma, ubicar la dependencia actual de `hnswlib`:

```
hnswlib = { version = "0.8.2", git = "https://github.com/chroma-core/hnswlib.git", branch = "master" }
```

----------

Reemplazarla por la versión local:

```
hnswlib = { path = "../hnswlib" }
```

----------

### 4. Compilación de Chroma

Compilar nuevamente el proyecto para aplicar la versión modificada del índice:

```
cargo build --release
```

## Autor

María de los Angeles Vásquez Pineda  - Curso de Estructuras de Datos Avanzadas