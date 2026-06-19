# Dionisio RevOps Agent

Copiloto interno para o case tecnico da Dionisio, usando OpenAI Agents SDK com OpenRouter e um executor deterministico para a API mockada.

## Setup

```powershell
python -m pip install -e ".[dev]"
$env:OPENROUTER_API_KEY = "..."
$env:DIONISIO_API_KEY = "key-marcelo-maia-fernandes-filho"
$env:OPENROUTER_MODEL = "deepseek/deepseek-v4-flash"
```

Opcional:

```powershell
$env:OPENROUTER_SITE_URL = "https://github.com/seu-usuario/dionisio"
$env:OPENROUTER_APP_TITLE = "Dionisio RevOps Agent"
$env:OPENAI_AGENTS_TRACING = "false"
$env:DIONISIO_INTERACTIVE_APPROVAL = "true"
```

## Rodar

Documentacao tecnica:

- [Arquitetura deep dive](docs/architecture_deep_dive.md)
- [Referencia completa do codigo em src/dionisio_agent](docs/source_reference.md)

```powershell
dionisio-agent ask "Quantas reservas temos para hoje a noite e quantos lugares sobram?"
```

Abrir a interface web de chat:

```powershell
uvicorn dionisio_agent.web:app --host 0.0.0.0 --port 8080
```

Depois acesse:

```text
http://localhost:8080/
```

A interface usa `/api/chat/stream` para renderizar deltas da resposta assim que
o Agents SDK começa a emitir texto. O endpoint JSON `/api/chat` continua
disponivel como fallback.

Abrir uma conversa interativa com memoria persistente:

```powershell
dionisio-agent chat --session-id teste-case
```

Use o mesmo `--session-id` para continuar a conversa em outra execucao. As mensagens ficam em `.dionisio_agent/sessions.sqlite`.

Listar operacoes conhecidas pelo catalogo:

```powershell
dionisio-agent operations --query reservas
```

Rodar testes:

```powershell
pytest
```

Rodar testes reais do agente contra OpenRouter + API Dionisio:

```powershell
$env:RUN_LIVE_AGENT_TESTS = "1"
pytest tests/test_live_agent.py
```

Rodar evals em modo checklist:

```powershell
dionisio-evals
```

Rodar evals ao vivo exige `OPENROUTER_API_KEY` e `DIONISIO_API_KEY`:

```powershell
dionisio-evals --live
```

## GraphRAG / Neo4j

O projeto suporta uma knowledge base semantica em Neo4j com `neo4j-graphrag`.
Ela e opcional e read-only para o agent: serve para explicar a API, relacionar
dominios, recuperar workflows ordenados e melhorar a selecao de operacoes, mas
nao executa chamadas HTTP. O agente nao usa GraphRAG como primeira fonte:
ele deve consultar `search_api_operations` e usar `search_api_knowledge` apenas
como ultimo recurso, quando o catalogo OpenAPI nao for suficiente para entender
o workflow, relacionar dominios ou escolher uma resposta segura. O ingest cria
embeddings com `openai/text-embedding-3-small` via OpenRouter em um indice
vetorial unico `KnowledgeNode`.

Ingerir o OpenAPI no Neo4j:

```powershell
dionisio-graphrag-ingest --clear
```

Auditar no Neo4j Browser:

```cypher
MATCH (n)-[r]->(m)
RETURN n, r, m
LIMIT 100
```

Detalhes: [GraphRAG com Neo4j](docs/graphrag_neo4j.md).

## Variaveis

- `OPENROUTER_API_KEY`: chave usada pelo agente e pelos embeddings do GraphRAG.
- `OPENROUTER_BASE_URL`: default `https://openrouter.ai/api/v1`.
- `OPENROUTER_MODEL`: default `deepseek/deepseek-v4-flash`; escolha um modelo com tool calling.
- `OPENAI_AGENTS_TRACING`: default `false`; mantenha assim com OpenRouter, ou use `true` se tambem configurar `OPENAI_API_KEY` para exportar traces.
- `OPENAI_API_KEY`: fallback opcional para embeddings diretos fora do OpenRouter.
- `DIONISIO_API_KEY`: chave do case.
- `DIONISIO_API_BASE_URL`: default `https://dionisio-crm.web.app`.
- `DIONISIO_OPENAPI_URL`: default `https://app.odionisio.com/api/case-mock/docs.json`.
- `DIONISIO_AGENT_MAX_TURNS`: default `25`; limite de rodadas modelo/tool por resposta do agente.
- `DIONISIO_AGENT_MAX_OUTPUT_TOKENS`: default `1200`; limite de tokens de saida para evitar respostas longas e reduzir latencia no chat.
- `DIONISIO_APPROVAL_POLICY`: `destructive` ou `mutations`.
- `DIONISIO_INTERACTIVE_APPROVAL`: quando `true`, tools podem pedir confirmacao no terminal.
- `DIONISIO_GRAPHRAG_ENABLED`: `true` para expor `search_api_knowledge`.
- `DIONISIO_GRAPHRAG_CONFIDENCE_THRESHOLD`: default `0.75`; limite para considerar GraphRAG como ultimo recurso, sem substituir o fluxo principal por `search_api_operations`.
- `DIONISIO_GRAPHRAG_CACHE_TTL_SECONDS`: default `300`; cache em memoria para buscas repetidas de `search_api_knowledge`.
- `DIONISIO_GRAPHRAG_CACHE_MAX_ENTRIES`: default `128`; limite de entradas do cache GraphRAG por runtime.
- `DIONISIO_GRAPHRAG_EFFECTIVE_SEARCH_RATIO`: default `2`; controla a amplitude da busca vetorial no Neo4j, preservando `top_k` mas reduzindo candidatos extras.
- `DIONISIO_GRAPHRAG_EMBEDDING_TIMEOUT_SECONDS`: default `6`; timeout do cliente de embeddings usado pela busca semantica.
- `DIONISIO_GRAPHRAG_NEO4J_TIMEOUT_SECONDS`: default `3`; timeout de conexao/leitura para consultas Neo4j do GraphRAG.
- `NEO4J_URI`: URI Bolt do Neo4j, por exemplo `bolt://host:7687`; dentro da Railway, prefira `bolt://${{dionisio-neo4j.RAILWAY_PRIVATE_DOMAIN}}:7687`.
- `NEO4J_USERNAME`: default `neo4j`.
- `NEO4J_PASSWORD`: senha do Neo4j.
- `NEO4J_DATABASE`: default `neo4j`.
- `WEB_CHAT_SESSION_DB_PATH`: default `.dionisio_agent/web_sessions.sqlite`.
- O chat web nao aplica timeout de agente na aplicacao; ele aguarda o `Runner.run` concluir ou falhar.
- `WEB_CHAT_SESSION_HISTORY_LIMIT`: default `0` (sem limite); limite de itens recentes carregados no contexto do agente por conversa. Use um valor alto se precisar limitar, porque turnos com tools geram varios itens internos.
