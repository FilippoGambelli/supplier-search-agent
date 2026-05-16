# Supplier Search Agent

AI-powered supplier discovery and contact extraction tool for construction and procurement workflows.

## Setup

1. Activate the virtual environment (Windows PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

2. Run Services

Start Docker containers:

```bash
docker compose up -d
```

3. Run the Backend

Start the FastAPI development server:

```bash
uvicorn src.api.routes:app --reload --port 8000
```

```bash
python .\src\main.py
```

```bash
pip install -e .
```


```bash
langgraph dev
```

Ecco una versione **breve, pulita e in inglese** da incollare nel tuo `README.md`:

---

## Database (PostgreSQL + pgAdmin)

The project uses PostgreSQL with a web UI via pgAdmin.

## pgAdmin UI

Open:

```
http://localhost:5050
```

Login:

* Email: `admin@admin.com`
* Password: `admin`

---

## Connect database

In pgAdmin → Register Server:

* Host: `postgres`
* Port: `5432`
* DB: `suppliersearchagentdb`
* User: `admin`
* Password: `admin`