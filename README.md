# Running locally (after a reboot)

## 1. Start MySQL

MySQL runs as a Windows service (`MySQL80`). Check it's running, start it if not:

```powershell
Get-Service MySQL80
Start-Service MySQL80   # only if not already Running
```

## 2. Start the Django backend

```bash
cd backend
python manage.py runserver 8000
```

Requires `backend/.env` to exist with `MYSQL_DATABASE`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_HOST`, `MYSQL_PORT`.

## 3. Start the Next.js frontend

```bash
cd frontend
npm run dev
```

## 4. Open the app

**http://localhost:3000** — not `127.0.0.1:3000`. Session cookies are scoped to the `localhost` hostname; visiting via `127.0.0.1` breaks login/session persistence.
