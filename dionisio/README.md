# Como executar a aplicacao do zero

## 1. Instale os pre-requisitos

Instale:

- Python 3.11 ou superior
- Git

Confira no terminal:

```powershell
python --version
git --version
```

## 2. Baixe o projeto

```powershell
git clone <URL_DO_REPOSITORIO>
cd dionisio
```

Se o projeto ja estiver baixado, entre na pasta dele:

```powershell
cd C:\Users\marce\OneDrive\Documentos\GitHub\dionisio-marcelo-maia\dionisio
```

## 3. Crie e ative o ambiente virtual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Se o PowerShell bloquear a ativacao, rode:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 4. Instale a aplicacao

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## 5. Crie o arquivo .env

Na raiz do projeto, crie um arquivo chamado `.env`:

```powershell
New-Item -ItemType File -Name .env
```

Abra o `.env` e cole exatamente este conteudo:

```env
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=
OPENROUTER_MODEL=



OPENAI_AGENTS_TRACING=false

# Dionisio case API
DIONISIO_API_KEY=key-marcelo-maia-fernandes-filho
DIONISIO_API_BASE_URL=https://dionisio-crm.web.app
DIONISIO_OPENAPI_URL=https://app.odionisio.com/api/case-mock/docs.json
DIONISIO_TIMEOUT_SECONDS=


DIONISIO_APPROVAL_POLICY=destructive


DIONISIO_INTERACTIVE_APPROVAL=false


DIONISIO_GRAPHRAG_ENABLED=true
NEO4J_URI=
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=
NEO4J_DATABASE=neo4j
DIONISIO_GRAPHRAG_ENABLED=true
```

Depois preencha pelo menos:

```env
OPENROUTER_API_KEY=sua_chave_do_openrouter
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=deepseek/deepseek-v4-flash
OPENROUTER_SITE_URL=http://localhost:8080
OPENROUTER_APP_TITLE=Dionisio RevOps Agent
DIONISIO_TIMEOUT_SECONDS=12
```

## 6. Teste a instalacao

Com o ambiente virtual ativado, rode:

```powershell
dionisio-agent operations --query reservas
```

Se o comando retornar uma lista de operacoes em JSON, a aplicacao conseguiu ler a documentacao da API Dionisio.

## 7. Teste o agente no terminal

```powershell
dionisio-agent chat --session-id teste-local
```

## 8. Execute a aplicacao web

```powershell
dionisio-web
```

Acesse no navegador:

```text
http://localhost:8080/
```

## 9. Execute com uvicorn, se preferir

```powershell
python -m uvicorn dionisio_agent.web:app --host 127.0.0.1 --port 8080
```

Acesse:

```text
http://localhost:8080/
```

## 10. Rode os testes

```powershell
pytest
```

## 11. Rode as avaliacoes

```powershell
dionisio-evals
```

## 12. Atualize o GraphRAG, se necessario

Se precisar recriar a base GraphRAG no Neo4j usando a documentacao atual da API, rode:

```powershell
dionisio-graphrag-ingest --clear
```
