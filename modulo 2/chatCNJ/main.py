from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# Importa os routers dos módulos
from app.modules.routes import router as chatnormas_router

# Cria a aplicação principal
app = FastAPI(
    title="Programa de Mestrado em Modelagem computacional e transformação digital",
    description="Projeto construído em programa de Mestrado de Modelagem Computacional"
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monta o diretório de arquivos estáticos (CSS, JS, Imagens)
app.mount("/static", StaticFiles(directory="app/frontend"), name="static")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "app/templates"))

# --- Inclusão das Rotas das APIs ---
app.include_router(chatnormas_router, prefix="/chatnormas", tags=["Chat Normas"])

# --- Rotas para servir as Páginas HTML ---
@app.get("/chatnormas", response_class=HTMLResponse, tags=["Frontend Pages"])
async def read_chatnormas(request: Request):
    """Serve a página do Chat chatnormas."""
    return templates.TemplateResponse("chatnormas.html", {"request": request, "active_page": "chatnormas"})