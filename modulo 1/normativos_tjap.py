"""
Pipeline completo: Leitura de atos normativos do Postgres, limpeza HTML,
chunking inteligente, geração de embeddings com BGE-M3 e gravação em pgvector.
"""

import os
import re
import json
import html
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
import numpy as np
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoModel

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

# Banco de dados fonte (leitura)
DB_SETTINGS_SOURCE = {
    "host": "10.10.0.13",
    "port": 5454,
    "user": "web_r",
    "password": "1234",
    "dbname": "dbgeral"
}

# Banco de dados destino (escrita - pgvector)
DB_SETTINGS_TARGET = {
    "host": "localhost",  # AJUSTAR CONFORME SEU AMBIENTE
    "port": 5433,
    "user": "postgres",
    "password": "postgres",
    "dbname": "normativos"
}

DB_SCHEMA = "public"
DB_TABLE = "normativos_tjap_bgem3"

# Modelo de embedding
TARGET_MODEL = {
    "name": "BAAI/bge-m3",
    "dimension": 1024,
    "short_name": "bgem3"
}

# Parâmetros de processamento
CHUNK_TARGET_TOKENS = 350
CHUNK_TARGET_CHARS = 1800
CHUNK_OVERLAP_PERCENT = 0.12

MAX_WORKERS = 2
DB_BATCH_SIZE = 150
PROCESSING_MEGA_BATCH_SIZE = 500

ENCODE_BATCH_SIZE = 8

# Checkpoint
CHECKPOINT_FILE = "./scraping_checkpoint_tjap.json"
CHECKPOINT_ENABLED = True

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class Document:
    """Representa um ato normativo."""
    id: int
    ementa: str
    texto: str
    situacao: Optional[str]
    tipo: Optional[str]
    origem: Optional[str]
    tema: Optional[str]
    fonte: Optional[str]
    inicio_vigencia: Optional[str]
    fim_vigencia: Optional[str]
    data_publicacao: Optional[str]
    data_assinatura: Optional[str]

@dataclass
class Chunk:
    """Representa um chunk de texto."""
    document: str
    metadata: Dict[str, Any]
    source_id: int
    chunk_index: int

# ============================================================================
# CHECKPOINT
# ============================================================================

class CheckpointManager:
    """Gerencia checkpoints para retomar processamento."""
    
    def __init__(self, filepath: str, enabled: bool = True):
        self.filepath = filepath
        self.enabled = enabled
        self.data = self._load()
    
    def _load(self) -> Dict:
        """Carrega checkpoint do disco."""
        if not self.enabled or not os.path.exists(self.filepath):
            return {"processed_ids": [], "last_update": None}
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Erro ao carregar checkpoint: {e}")
            return {"processed_ids": [], "last_update": None}
    
    def save(self, processed_ids: List[int]):
        """Salva checkpoint no disco."""
        if not self.enabled:
            return
        try:
            self.data["processed_ids"] = list(set(self.data["processed_ids"] + processed_ids))
            self.data["last_update"] = datetime.now().isoformat()
            with open(self.filepath, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logger.error(f"Erro ao salvar checkpoint: {e}")
    
    def is_processed(self, doc_id: int) -> bool:
        """Verifica se documento já foi processado."""
        return doc_id in self.data["processed_ids"]
    
    def get_processed_count(self) -> int:
        """Retorna quantidade de documentos processados."""
        return len(self.data["processed_ids"])

# ============================================================================
# LIMPEZA DE HTML
# ============================================================================

class HTMLCleaner:
    """Limpa HTML preservando estrutura textual."""
    
    @staticmethod
    def clean(html_text: str) -> str:
        """
        Limpa HTML removendo tags, estilos e lixo, preservando parágrafos.
        """
        if not html_text or not html_text.strip():
            return ""
        
        # Decodificar entidades HTML
        text = html.unescape(html_text)
        
        # Remover scripts e styles
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Substituir tags de quebra por marcadores temporários
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</p>', '\n\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</div>', '\n\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</h[1-6]>', '\n\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</li>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</tr>', '\n', text, flags=re.IGNORECASE)
        
        # Remover todas as tags HTML restantes
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        
        # Remover múltiplos espaços em branco
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remover linhas vazias múltiplas (máximo 2 quebras consecutivas)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remover espaços no início/fim de linhas
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Remover espaços antes de pontuação
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        
        # Normalizar aspas e travessões
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('–', '-').replace('—', '-')
        
        # Remover caracteres de controle
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        
        return text.strip()

# ============================================================================
# CHUNKING
# ============================================================================

class TextChunker:
    """Divide texto em chunks com overlap inteligente."""
    
    def __init__(self, target_chars: int = CHUNK_TARGET_CHARS, overlap: float = CHUNK_OVERLAP_PERCENT):
        self.target_chars = target_chars
        self.overlap = overlap
        self.overlap_chars = int(target_chars * overlap)
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Divide texto em sentenças."""
        # Padrão para detectar fim de sentença
        pattern = r'(?<=[.!?])\s+(?=[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ])'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk(self, doc: Document) -> List[Chunk]:
        """
        Divide documento em chunks preservando sentenças.
        """
        text = doc.texto
        if not text or len(text.strip()) < 100:
            return []
        
        sentences = self.split_into_sentences(text)
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for i, sentence in enumerate(sentences):
            sentence_len = len(sentence)
            
            # Se adicionar esta sentença ultrapassar o limite
            if current_length + sentence_len > self.target_chars and current_chunk:
                # Criar chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(self._create_chunk(doc, chunk_text, chunk_index, len(chunks)))
                chunk_index += 1
                
                # Calcular overlap: pegar últimas sentenças até atingir overlap_chars
                overlap_sents = []
                overlap_len = 0
                for sent in reversed(current_chunk):
                    if overlap_len + len(sent) <= self.overlap_chars:
                        overlap_sents.insert(0, sent)
                        overlap_len += len(sent)
                    else:
                        break
                
                # Reiniciar chunk com overlap
                current_chunk = overlap_sents
                current_length = overlap_len
            
            # Adicionar sentença ao chunk atual
            current_chunk.append(sentence)
            current_length += sentence_len
        
        # Adicionar último chunk se não estiver vazio
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(self._create_chunk(doc, chunk_text, chunk_index, len(chunks)))
        
        return chunks
    
    def _create_chunk(self, doc: Document, text: str, index: int, total: int) -> Chunk:
        """Cria objeto Chunk com metadados."""
        metadata = {
            "source_id": doc.id,
            "chunk_index": index,
            "total_chunks": total,
            "ementa": doc.ementa,
            "situacao": doc.situacao,
            "tipo": doc.tipo,
            "origem": doc.origem,
            "tema": doc.tema,
            "fonte": doc.fonte,
            "inicio_vigencia": str(doc.inicio_vigencia) if doc.inicio_vigencia else None,
            "fim_vigencia": str(doc.fim_vigencia) if doc.fim_vigencia else None,
            "data_publicacao": str(doc.data_publicacao) if doc.data_publicacao else None,
            "data_assinatura": str(doc.data_assinatura) if doc.data_assinatura else None,
            "char_count": len(text),
            "chunk_start": index * self.target_chars,
            "chunk_end": (index + 1) * self.target_chars
        }
        
        return Chunk(
            document=text,
            metadata=metadata,
            source_id=doc.id,
            chunk_index=index
        )

# ============================================================================
# EMBEDDINGS
# ============================================================================

class EmbeddingGenerator:
    """Gera embeddings usando BAAI/bge-m3."""
    
    def __init__(self, model_name: str = TARGET_MODEL["name"], batch_size: int = ENCODE_BATCH_SIZE):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Carregando modelo {model_name} em {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Modelo carregado com sucesso")
    
    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling considerando attention mask."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Gera embeddings normalizados para lista de textos.
        """
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenizar
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Mover para device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Gerar embeddings
            with torch.no_grad():
                model_output = self.model(**encoded)
                embeddings = self.mean_pooling(model_output, encoded['attention_mask'])
                
                # Normalizar
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)

# ============================================================================
# DATABASE
# ============================================================================

class DatabaseManager:
    """Gerencia conexões e operações no banco de dados."""
    
    @staticmethod
    def fetch_documents(checkpoint: CheckpointManager) -> List[Document]:
        """Busca documentos do banco fonte."""
        query = """
        SELECT ato.id, ato.ementa, doc.teor as texto, ato.situacao, ato.tipo, ato.origem, ato.tema, ato.fonte,
               ato.inicio_vigencia, ato.fim_vigencia, ato.data_publicacao, ato.data_assinatura
        FROM ato_normativo.sig_vw_ato_normativo ato
        LEFT JOIN ato_normativo.doc_ato doc ON ato.id = doc.id
        WHERE ato.publicado = 't' AND ato.rev_1
        ORDER BY ato.id ASC;
        """
        
        logger.info("Conectando ao banco fonte...")
        conn = psycopg2.connect(**DB_SETTINGS_SOURCE)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        logger.info("Executando consulta...")
        cursor.execute(query)
        rows = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        logger.info(f"Total de documentos recuperados: {len(rows)}")
        
        # Filtrar documentos já processados
        documents = []
        skipped = 0
        for row in rows:
            if checkpoint.is_processed(row['id']):
                skipped += 1
                continue
            documents.append(Document(**row))
        
        logger.info(f"Documentos já processados (pulados): {skipped}")
        logger.info(f"Documentos a processar: {len(documents)}")
        
        return documents
    
    @staticmethod
    def initialize_target_table():
        """Cria tabela no banco destino se não existir."""
        conn = psycopg2.connect(**DB_SETTINGS_TARGET)
        cursor = conn.cursor()
        
        # Criar extensão pgvector se não existir
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Criar tabela
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {DB_SCHEMA}.{DB_TABLE} (
            id BIGSERIAL PRIMARY KEY,
            document TEXT NOT NULL,
            metadata JSONB,
            embedding vector({TARGET_MODEL['dimension']})
        );
        """
        cursor.execute(create_table_query)
        
        # Criar índices
        cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{DB_TABLE}_metadata_source 
        ON {DB_SCHEMA}.{DB_TABLE} USING gin ((metadata->'source_id'));
        """)
        
        cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{DB_TABLE}_embedding 
        ON {DB_SCHEMA}.{DB_TABLE} USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("Tabela destino inicializada")
    
    @staticmethod
    def upsert_chunks(chunks: List[Chunk], embeddings: np.ndarray):
        """Insere ou atualiza chunks no banco destino."""
        conn = psycopg2.connect(**DB_SETTINGS_TARGET)
        cursor = conn.cursor()
        
        # Preparar dados
        data = []
        for chunk, embedding in zip(chunks, embeddings):
            data.append((
                chunk.document,
                json.dumps(chunk.metadata),
                embedding.tolist()
            ))
        
        # UPSERT
        query = f"""
        INSERT INTO {DB_SCHEMA}.{DB_TABLE} (document, metadata, embedding)
        VALUES %s
        ON CONFLICT DO NOTHING;
        """
        
        execute_values(cursor, query, data)
        conn.commit()
        
        inserted = cursor.rowcount
        cursor.close()
        conn.close()
        
        return inserted

# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

class Pipeline:
    """Pipeline completo de processamento."""
    
    def __init__(self):
        self.checkpoint = CheckpointManager(CHECKPOINT_FILE, CHECKPOINT_ENABLED)
        self.cleaner = HTMLCleaner()
        self.chunker = TextChunker()
        self.embedder = EmbeddingGenerator()
    
    def process_document(self, doc: Document) -> List[Chunk]:
        """Processa um documento: limpa e cria chunks."""
        # Limpar HTML
        clean_text = self.cleaner.clean(doc.texto)
        
        if not clean_text or len(clean_text) < 100:
            logger.warning(f"Documento {doc.id} muito curto após limpeza")
            return []
        
        # Atualizar documento com texto limpo
        doc.texto = clean_text
        
        # Criar chunks
        chunks = self.chunker.chunk(doc)
        
        return chunks
    
    def process_batch(self, documents: List[Document]) -> int:
        """Processa lote de documentos."""
        all_chunks = []
        processed_ids = []
        
        logger.info(f"Processando {len(documents)} documentos...")
        
        # Processar documentos em paralelo
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_doc = {executor.submit(self.process_document, doc): doc for doc in documents}
            
            for future in as_completed(future_to_doc):
                doc = future_to_doc[future]
                try:
                    chunks = future.result()
                    if chunks:
                        all_chunks.extend(chunks)
                        processed_ids.append(doc.id)
                except Exception as e:
                    logger.error(f"Erro ao processar documento {doc.id}: {e}")
        
        if not all_chunks:
            logger.warning("Nenhum chunk gerado neste lote")
            return 0
        
        logger.info(f"Total de chunks gerados: {len(all_chunks)}")
        
        # Gerar embeddings
        logger.info("Gerando embeddings...")
        texts = [chunk.document for chunk in all_chunks]
        embeddings = self.embedder.encode(texts)
        
        # Inserir no banco em lotes
        logger.info("Inserindo no banco de dados...")
        total_inserted = 0
        
        for i in range(0, len(all_chunks), DB_BATCH_SIZE):
            batch_chunks = all_chunks[i:i + DB_BATCH_SIZE]
            batch_embeddings = embeddings[i:i + DB_BATCH_SIZE]
            
            inserted = DatabaseManager.upsert_chunks(batch_chunks, batch_embeddings)
            total_inserted += inserted
        
        # Salvar checkpoint
        self.checkpoint.save(processed_ids)
        
        logger.info(f"Lote processado: {total_inserted} chunks inseridos")
        
        return total_inserted
    
    def run(self):
        """Executa pipeline completo."""
        logger.info("=" * 80)
        logger.info("INICIANDO PIPELINE DE PROCESSAMENTO")
        logger.info("=" * 80)
        
        # Inicializar tabela destino
        DatabaseManager.initialize_target_table()
        
        # Buscar documentos
        documents = DatabaseManager.fetch_documents(self.checkpoint)
        
        if not documents:
            logger.info("Nenhum documento novo para processar")
            return
        
        # Processar em mega-batches
        total_processed = 0
        
        for i in range(0, len(documents), PROCESSING_MEGA_BATCH_SIZE):
            batch = documents[i:i + PROCESSING_MEGA_BATCH_SIZE]
            logger.info(f"\n{'=' * 80}")
            logger.info(f"MEGA-BATCH {i//PROCESSING_MEGA_BATCH_SIZE + 1}: {len(batch)} documentos")
            logger.info(f"{'=' * 80}\n")
            
            inserted = self.process_batch(batch)
            total_processed += inserted
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE CONCLUÍDO")
        logger.info(f"Total de chunks processados e inseridos: {total_processed}")
        logger.info(f"Total de documentos processados: {self.checkpoint.get_processed_count()}")
        logger.info("=" * 80)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        pipeline = Pipeline()
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("\nProcessamento interrompido pelo usuário")
    except Exception as e:
        logger.error(f"Erro fatal: {e}", exc_info=True)