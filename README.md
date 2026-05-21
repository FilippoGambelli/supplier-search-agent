# Supplier Search Agent

AI-powered supplier discovery and contact extraction tool for construction and procurement workflows.

## Setup

1. Activate the virtual environment (Windows PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

2. Run Services

```bash
docker compose up -d
```

```bash
python .\src\main.py
```

```bash
langgraph dev
```

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

`docker compose down`
`docker volume rm supplier-search-agent_postgres_data`