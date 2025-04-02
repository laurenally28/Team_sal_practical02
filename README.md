# Team_sal_practical02
In this project, you and your team will build a local Retrieval-Augmented Generation system that allows a user to query the collective DS4300 notes from members of your team. 

## Source Code
- `pipeline.py` - Processes PDFs by extracting, preprocessing, and embedding text into a vector database for retrieval-augmented generation (RAG)-based search. It runs experiments with different configurations, benchmarks query performance, and logs results, making it useful for evaluating embeddings and vector database performance. it also has the capability to recreate the interactive search method utilized on the midterm exam. 
- `src/embeddings.py` - Generates text embeddings using a chosen embedding model. It dynamically selects the embedding method based on the specified model, allowing flexibility in embedding generation for downstream tasks.
- `src/vector_store.py` - generates an instance of either Redis or Chroma vector database. This vector store is then applied accross all index and serch instance for that specific pipeline execution. 
- `src/llm_search.py` - generates the prompt template obhect to be passed into the RAG. Also intializes and interactive search instance if specified in the pipeline execution. 
- `src/preprocess_text.py` -  preprocesses the input text for embedding and vector storage.
- `analysis.ipynb` - notebook containing all relevant figures created from data in `results/experiment_results.csv`

## Pipeline Execution

### 1. Create/Activate Environment
```
conda create -n practical_02_env
conda activate practical_02_env
```

### 2. Install Requirements
```
pip install -r requirements.txt
```

### 3. Navigate to Cloned Repo Location in Terminal
```
cd ../path/to/Team_sal_practical02
```

### 4. Execute Pipeline

#### Mode 1: Running Experiments (Default)
To generate experiment data across different configurations:
```
python pipeline.py --mode experiments
```

You can also specify the LLM model to use (default is `llama`):
```
python pipeline.py --mode experiments --model mistral
```
Options for `--model`:
- `llama` (default)
- `mistral`

#### Mode 2: Interactive Search
To run an interactive RAG-based search:
```
python pipeline.py --mode interactive
```
You can specify the model as well:
```
python pipeline.py --mode interactive --model mistral
```

### Available CLI Arguments
- `--mode`: Execution mode (`experiments` or `interactive`). Default is `experiments`.
- `--model`: Selects which LLM to use (`llama` or `mistral`). Default is `llama`.

### Example Commands
- Run experiments with default settings:
  ```
  python pipeline.py
  ```
- Run experiments using Mistral:
  ```
  python pipeline.py --mode experiments --model mistral
  ```
- Start an interactive search with Llama:
  ```
  python pipeline.py --mode interactive --model llama
  ```

