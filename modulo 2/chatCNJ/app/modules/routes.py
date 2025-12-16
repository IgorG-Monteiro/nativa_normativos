from fastapi import APIRouter, HTTPException 
from pydantic import BaseModel
import httpx
import logging
import json

# Configurar logging para ver as mensagens de depuração
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# DEV
N8N_WEBHOOK_URL = "http://n8n_server:5678/webhook/66606d14-c81d-4523-ac00-8238854e560c"
# PROD
# N8N_WEBHOOK_URL = "http://157.173.125.173:5678/webhook/1498ff44-3cc2-4ef9-bfab-ea8416eeb00b"

class ChatRequest(BaseModel):
    message: str
    sessionId: str
    db: str  # Adicionado campo db (aceita "cnj" ou "tjap")

async def log_request(request):
    logger.info("================= REQUISIÇÃO SAINDO DO FASTAPI PARA O N8N =================")
    logger.info(f"Método: {request.method} {request.url}")
    logger.info("Cabeçalhos:")
    for key, value in request.headers.items():
        logger.info(f"  {key}: {value}")
    
    await request.aread() 
    body_bytes = request.content
    logger.info("Corpo da Requisição (como texto):")
    logger.info(body_bytes.decode('utf-8'))
    logger.info("=========================================================================")

@router.post("/legislacao")
async def chat_response(request_data: ChatRequest):
    logger.info(f"Recebida mensagem da sessão: {request_data.sessionId}")
    logger.info(f"Banco de dados selecionado: {request_data.db}")
    
    # Validação do campo db
    if request_data.db not in ["cnj", "tjap"]:
        logger.warning(f"Banco de dados inválido recebido: {request_data.db}. Usando 'cnj' como padrão.")
        db_value = "cnj"
    else:
        db_value = request_data.db
    
    # Payload enviado para o N8N agora inclui o campo db
    n8n_payload = {
        "texto": request_data.message,
        "sessionId": request_data.sessionId,
        "db": db_value  # Envia o banco selecionado para o N8N
    }
    
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "insomnia/11.5.0",
    }
    
    payload_bytes = json.dumps(n8n_payload).encode('utf-8')
    
    try:
        async with httpx.AsyncClient(timeout=None, event_hooks={'request': [log_request]}) as client:
            
            response = await client.post(N8N_WEBHOOK_URL, content=payload_bytes, headers=headers)
            
            response.raise_for_status()
            
            logger.info(f"Resposta recebida com sucesso do N8N (Status: {response.status_code})")
            
            # Log da resposta completa do N8N
            n8n_response = response.json()
            logger.info("================= RESPOSTA DO N8N =================")
            logger.info(f"Tipo da resposta: {type(n8n_response)}")
            logger.info(f"Conteúdo completo: {json.dumps(n8n_response, indent=2, ensure_ascii=False)}")
            logger.info("===================================================")
            
            # O N8N retorna uma lista com um objeto dentro, então pegamos o primeiro elemento
            if isinstance(n8n_response, list) and len(n8n_response) > 0:
                logger.info("N8N retornou uma lista. Extraindo o primeiro elemento...")
                return n8n_response[0]
            
            return n8n_response
            
    except httpx.RequestError as e:
        logger.error(f"ERRO DE CONEXÃO com o N8N em {N8N_WEBHOOK_URL}: {e}")
        raise HTTPException(
            status_code=503, 
            detail="Não foi possível conectar ao serviço de IA (N8N)."
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"O N8N RETORNOU UM ERRO HTTP: {e.response.status_code}")
        logger.error(f"Resposta do N8N: {e.response.text}")
        raise HTTPException(
            status_code=502, 
            detail=f"O serviço de IA (N8N) retornou uma resposta inválida."
        )
    except Exception as e:
        logger.error(f"Ocorreu um erro inesperado: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erro interno do servidor.")

@router.get("/test_chat")
async def test_route_chat():
    logger.info("ROTA DE TESTE DO CHAT ACIONADA!")
    return {"status": "ok", "message": "O router do Chatnormas está a funcionar!"}