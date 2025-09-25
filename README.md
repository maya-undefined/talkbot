# Talkbot

talkbot is a simple chatbot pipeline that shows the workings of RAG 
pipelines.

Users typically can start by uploading files relevant to a query. The 
raw text file is saved. Then the document is split into chunks. Those
chunks are then sent to openai, and the embeddings are returned.

The openai embeddings are then stored in the local pgvector db. 

Upon a chat message being received, openai will be contacted to retrieve 
embeddings for this message. talkbot will then search the local pgvector 
database. The comparison used is pgvector's cosine simliarity.

After relevant documents from the pgvector are retrieved, they are 
combined to the user's message as data. Then the combined message is
sent to openai to be answered, and retrieved.

Local

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    uvicorn app.main:app --reload --port 8000 --env-file .env

Container

    docker build -t fin-advisor:dev .
    docker run --rm -p 8000:8000 --env-file .env.example fin-advisor:dev

#### Personas

Alter the app/core/personas.py file to suit your needs.

### Environment variables

Consult the env.example. You should have something like

    OPENAI_API_KEY=
    TENANT_ID=t1
    LOG_LEVEL=INFO
    DATABASE_URL=postgresql+psycopg://appuser:pass@localhost:5432/finadvisor
    OPENAI_BASE_URL=https://api.openai.com/v1


### Gradio UI

To use, start in another terminal window:

    python gradio_ui.py


### Commands to use

Chat with  the ai and ask

```
 curl -X POST http://127.0.0.1:8000/chat   -H "Content-Type: application/json"   -d '{
        "tenant_id":"t1",
        "persona_id":"budget_coach",
        "messages":[
          {"role":"user","content":"Summarize my restaurant spending in September."}
        ]
      }'
```

Upload a document to openai, obtain embeddings, upload embeddings to postgres
```
curl -F tenant_id=t1 -F file=@card.csv http://127.0.0.1:8000/ingest_pg
```

Clear out the DB
```
docker exec -it fin-pg psql -U postgres -c "
  SELECT pg_terminate_backend(pid)
  FROM pg_stat_activity
  WHERE datname = 'finadvisor' AND pid <> pg_backend_pid();
  DROP DATABASE IF EXISTS finadvisor;
  CREATE DATABASE finadvisor OWNER appuser;
"
```

Run migrations
```
docker exec -i fin-pg psql -U appuser -d finadvisor   -f /dev/stdin < migrations/0001_init_pgvector.sql
```

Clear out the entire DB manually
```
docker exec -it fin-pg psql -U appuser -d finadvisor -c "
  DROP INDEX IF EXISTS vectors_embed_idx;
  DROP TABLE IF EXISTS vectors;
  DROP TABLE IF EXISTS chunks;
  DROP TABLE IF EXISTS documents;
  -- Optional: remove extension (usually not needed)
  -- DROP EXTENSION IF EXISTS vector;
"
```

Blow away DB and restart for demo with `docker compose`
```
docker compose down
docker volume rm finance_bot_finpg_data
docker compose up -d 
```
