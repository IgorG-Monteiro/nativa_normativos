# -*- coding: utf-8 -*-
"""
ETL CNJ: Web Scraping → PostgreSQL (Versão Local)
Sistema de Checkpoint + Processamento em Múltiplos Chunks
"""

import os
import time
import json
import re
import unicodedata
import random
import torch
import psycopg2
import gc
import requests
from pathlib import Path
from typing import List, Dict
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from psycopg2 import sql
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

print("=" * 80)
print("⚙️  CONFIGURAÇÕES DO SISTEMA")
print("=" * 80)

# Configurações do Scraping
ROOT_URL = 'https://atos.cnj.jus.br'
LISTING_URL = ROOT_URL + '/atos?atos=sim&page={}'
PAGE_LIMIT = 0  # 0 = processar todas as páginas
LINK_LIMIT = 0  # 0 = processar todos os links
DELAY_ENABLED = True
DELAY_RANGE = (0.5, 1.5)

# Configurações de Otimização
MAX_WORKERS = 5
DB_BATCH_SIZE = 500
ENCODE_BATCH_SIZE = 32
PROCESSING_MEGA_BATCH_SIZE = 2000

# Sistema de Checkpoint
CHECKPOINT_FILE = Path("./scraping_checkpoint.json")
CHECKPOINT_ENABLED = True

# Configurações do Banco (ALTERE AQUI PARA SUAS CREDENCIAIS)
DB_SETTINGS = {
    "host": "localhost",  # ou IP do seu servidor
    "port": 5433,
    "user": "postgres",
    "password": "postgres",  # ⚠️ ALTERE AQUI
    "dbname": "normativos"
}
DB_SCHEMA = "public"
DB_TABLE = "normativos_cnj_bgem3"

# Configurações do Modelo
TARGET_MODEL = {
    "name": "BAAI/bge-m3",
    "dimension": 1024,
    "short_name": "bgem3"
}

# Detecção de Hardware
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"✓ Dispositivo: {DEVICE.upper()}")
print(f"✓ Modelo: {TARGET_MODEL['name']}")
print(f"✓ Tabela: {DB_SCHEMA}.{DB_TABLE}")
print(f"✓ Checkpoint: {'Habilitado' if CHECKPOINT_ENABLED else 'Desabilitado'}")

# ==============================================================================
# SISTEMA DE CHECKPOINT
# ==============================================================================

class CheckpointManager:
    """Gerencia o estado do scraping para permitir retomada"""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.state = self.load()

    def load(self) -> Dict:
        """Carrega estado salvo ou cria novo"""
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                print(f"\n✅ Checkpoint carregado!")
                print(f"   Última página processada: {state.get('last_page', 0)}")
                print(f"   Total de links processados: {state.get('total_links', 0)}")
                print(f"   Total de chunks inseridos: {state.get('total_chunks', 0)}")
                return state
            except Exception as e:
                print(f"\n⚠️  Erro ao carregar checkpoint: {e}")
                return self._new_state()
        else:
            print("\n📝 Nenhum checkpoint encontrado. Iniciando do zero.")
            return self._new_state()

    def _new_state(self) -> Dict:
        """Cria novo estado"""
        return {
            "last_page": 0,
            "total_links": 0,
            "total_chunks": 0,
            "processed_urls": [],
            "failed_urls": [],
            "start_time": time.time()
        }

    def save(self):
        """Salva estado atual"""
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def update(self, page: int = None, links: int = None, chunks: int = None,
               url: str = None, failed_url: str = None):
        """Atualiza estado"""
        if page is not None:
            self.state['last_page'] = page
        if links is not None:
            self.state['total_links'] += links
        if chunks is not None:
            self.state['total_chunks'] += chunks
        if url is not None and url not in self.state['processed_urls']:
            self.state['processed_urls'].append(url)
        if failed_url is not None:
            self.state['failed_urls'].append(failed_url)
        self.save()

    def is_processed(self, url: str) -> bool:
        """Verifica se URL já foi processada"""
        return url in self.state['processed_urls']

    def get_start_page(self) -> int:
        """Retorna página para iniciar/retomar"""
        return self.state['last_page'] + 1 if self.state['last_page'] > 0 else 1

# ==============================================================================
# FUNÇÕES DE PROCESSAMENTO HTML
# ==============================================================================

def slugify_column_name(name: str) -> str:
    """Normaliza nomes de colunas"""
    if not name:
        return "coluna_desconhecida"
    text = unicodedata.normalize('NFD', str(name))
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text).strip('_')
    return text if text else "coluna_desconhecida"

def parse_and_clean_html_content(content_soup: BeautifulSoup) -> str:
    """Remove tags e extrai texto limpo"""
    if not content_soup:
        return ""
    for tag in content_soup(['script', 'style', 'a', 'img']):
        tag.decompose()
    for p in content_soup.find_all('p'):
        p.replace_with(p.get_text() + '\n')
    cleaned_text = content_soup.get_text(separator=' ', strip=True)
    return cleaned_text.replace('\n ', '\n').replace(' \n', '\n')

def clean_legal_text(text: str) -> str:
    """Remove ruídos comuns em textos legais"""
    if not text:
        return ""
    text = re.sub(r'^\s*[\.]{5,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\.{5,}', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ==============================================================================
# ESTRATÉGIA DE CHUNKING
# ==============================================================================

def chunk_by_legal_structure(raw_text: str, document_id: str, **kwargs) -> List[Dict]:
    """
    Divide um normativo completo em múltiplos chunks baseados na estrutura legal
    """
    final_chunks = []

    # Parâmetros de Controle
    MAX_CHUNK_SIZE = 1500
    MIN_CHUNK_SIZE = 100
    RECURSIVE_CHUNK_SIZE = 800
    RECURSIVE_OVERLAP = 80

    if not raw_text or len(raw_text.strip()) < MIN_CHUNK_SIZE:
        print(f"  ⚠️  Texto muito curto ou vazio: {document_id[:80]}")
        return []

    # Regex para encontrar artigos
    article_pattern = re.compile(
        r'(?:^|\n)\s*(Art(?:igo)?\.?\s*\d+[º°]?\.?)',
        re.IGNORECASE | re.MULTILINE
    )

    matches = list(article_pattern.finditer(raw_text))

    print(f"  📄 Doc: {document_id[-30:]} | Tamanho: {len(raw_text)} chars | Artigos: {len(matches)}")

    # ESTRATÉGIA 1: Divide por artigos
    if len(matches) >= 2:
        # Preâmbulo
        preamble = raw_text[:matches[0].start()].strip()
        if len(preamble) > MIN_CHUNK_SIZE:
            if len(preamble) > MAX_CHUNK_SIZE:
                splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", ". ", " "],
                    chunk_size=RECURSIVE_CHUNK_SIZE,
                    chunk_overlap=RECURSIVE_OVERLAP,
                    length_function=len
                )
                preamble_chunks = splitter.split_text(preamble)
                for idx, p_chunk in enumerate(preamble_chunks):
                    if len(p_chunk.strip()) > MIN_CHUNK_SIZE:
                        final_chunks.append({
                            "document_id": document_id,
                            "autor": "Não identificado",
                            "tipo": "Preâmbulo",
                            "artigo_pai": f"Preâmbulo_parte_{idx+1}",
                            "chunk_text": p_chunk.strip()
                        })
            else:
                final_chunks.append({
                    "document_id": document_id,
                    "autor": "Não identificado",
                    "tipo": "Preâmbulo",
                    "artigo_pai": "Preâmbulo",
                    "chunk_text": preamble
                })

        # Processa cada artigo
        for i, match in enumerate(matches):
            article_marker = match.group(1).strip()
            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
            article_text = raw_text[start_pos:end_pos].strip()

            if len(article_text) <= MAX_CHUNK_SIZE:
                if len(article_text) > MIN_CHUNK_SIZE:
                    final_chunks.append({
                        "document_id": document_id,
                        "autor": "Não identificado",
                        "tipo": "Artigo",
                        "artigo_pai": article_marker,
                        "chunk_text": article_text
                    })
            else:
                splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n§", "\nParágrafo único", "\n§ 1", "\n§ 2",
                               "\nI -", "\nII -", "\na)", "\nb)", ".\n", ". ", " "],
                    chunk_size=RECURSIVE_CHUNK_SIZE,
                    chunk_overlap=RECURSIVE_OVERLAP,
                    length_function=len
                )
                sub_chunks = splitter.split_text(article_text)

                for j, sub_chunk in enumerate(sub_chunks):
                    if len(sub_chunk.strip()) > MIN_CHUNK_SIZE:
                        final_chunks.append({
                            "document_id": document_id,
                            "autor": "Não identificado",
                            "tipo": "Artigo (subdividido)",
                            "artigo_pai": f"{article_marker}_parte_{j+1}",
                            "chunk_text": sub_chunk.strip()
                        })

        print(f"    ✓ Gerou {len(final_chunks)} chunks por estrutura de artigos")

    # ESTRATÉGIA 2: Fallback - divisão recursiva
    else:
        print(f"    ⚠️  Poucos artigos ({len(matches)}). Usando estratégia recursiva.")

        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " "],
            chunk_size=RECURSIVE_CHUNK_SIZE,
            chunk_overlap=RECURSIVE_OVERLAP,
            length_function=len
        )
        chunks_text = splitter.split_text(raw_text)

        for i, chunk in enumerate(chunks_text):
            clean_chunk = chunk.strip()
            if len(clean_chunk) > MIN_CHUNK_SIZE:
                final_chunks.append({
                    "document_id": document_id,
                    "autor": "Não identificado",
                    "tipo": "Recursivo",
                    "artigo_pai": f"bloco_{i+1}",
                    "chunk_text": clean_chunk
                })

        print(f"    ✓ Gerou {len(final_chunks)} chunks por divisão recursiva")

    if len(final_chunks) == 0:
        print(f"    ❌ ERRO: Nenhum chunk gerado para {document_id[-50:]}")

    return final_chunks

# ==============================================================================
# PROCESSAMENTO DE LINKS
# ==============================================================================

def process_link(url: str, session: requests.Session, checkpoint: CheckpointManager) -> List[Dict]:
    """Baixa e processa uma URL completa de um normativo"""

    if checkpoint.is_processed(url):
        return []

    try:
        if DELAY_ENABLED:
            time.sleep(random.uniform(*DELAY_RANGE))

        response = session.get(url, timeout=30)
        response.raise_for_status()

        detailed_soup = BeautifulSoup(response.text, 'html.parser')
        main_div = detailed_soup.find('div', class_='geral_atos_normativos')

        if not main_div:
            main_div = detailed_soup.find('body')

        if not main_div:
            print(f"  ❌ Estrutura HTML não encontrada: {url[-50:]}")
            checkpoint.update(failed_url=url)
            return []

        # Extrai metadados e texto
        data_from_page = {}
        texto_completo = []

        for id_div in main_div.find_all('div', class_='identificacao'):
            column_name = id_div.get_text(strip=True)
            content_div = id_div.find_next_sibling('div')

            if content_div:
                safe_col_name = slugify_column_name(column_name)

                if safe_col_name == "texto":
                    cleaned_content = parse_and_clean_html_content(content_div)
                    data_from_page[safe_col_name] = cleaned_content
                    texto_completo.append(cleaned_content)
                else:
                    simple_text = content_div.get_text(strip=True)
                    data_from_page[safe_col_name] = simple_text

        if "texto" not in data_from_page or not data_from_page["texto"].strip():
            full_text_raw = parse_and_clean_html_content(main_div)
            texto_completo.append(full_text_raw)

        full_text = "\n\n".join(filter(None, texto_completo))

        if not full_text or len(full_text.strip()) < 100:
            print(f"  ❌ Texto insuficiente extraído: {url[-50:]}")
            checkpoint.update(failed_url=url)
            return []

        full_text = clean_legal_text(full_text)
        print(f"  📝 Extraído: {len(full_text)} caracteres de {url[-40:]}")

        # Gera chunks
        chunks = chunk_by_legal_structure(raw_text=full_text, document_id=url)

        if not chunks:
            print(f"  ❌ Nenhum chunk gerado: {url[-50:]}")
            checkpoint.update(failed_url=url)
            return []

        # Adiciona metadados
        metadata_sem_texto = data_from_page.copy()
        metadata_sem_texto.pop("texto", None)

        for chunk in chunks:
            chunk['metadata'] = metadata_sem_texto.copy()

        checkpoint.update(url=url)
        print(f"    ✅ {len(chunks)} chunks criados com sucesso")

        return chunks

    except Exception as e:
        print(f"  ❌ Erro ao processar {url[-50:]}: {str(e)[:100]}")
        checkpoint.update(failed_url=url)
        return []

# ==============================================================================
# FUNÇÕES DE BANCO DE DADOS
# ==============================================================================

def test_db_connection(db_settings: dict) -> bool:
    """Testa conexão"""
    try:
        conn = psycopg2.connect(**db_settings)
        cur = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        cur.close()
        conn.close()
        print(f"✅ Conexão OK")
        print(f"   {version.split(',')[0]}")
        return True
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def create_pg_table(conn, table_name: str, model_dimension: int):
    """Cria tabela com vetores"""
    with conn.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        create_table_sql = sql.SQL("""
            CREATE TABLE IF NOT EXISTS {} (
                id SERIAL PRIMARY KEY,
                document TEXT NOT NULL,
                metadata JSONB NOT NULL,
                embedding VECTOR({}) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """).format(
            sql.Identifier(table_name),
            sql.Literal(model_dimension)
        )
        cursor.execute(create_table_sql)

        index_name = f"{table_name}_embedding_idx"
        index_sql = sql.SQL("""
            CREATE INDEX IF NOT EXISTS {}
            ON {}
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """).format(
            sql.Identifier(index_name),
            sql.Identifier(table_name)
        )
        cursor.execute(index_sql)

        print(f"✓ Tabela '{table_name}' criada com sucesso")
        print(f"✓ Índice '{index_name}' criado com sucesso")

def batch_insert_data(conn, table_name: str, data_to_insert: List[tuple]):
    """Insere em batch"""
    if not data_to_insert:
        return
    insert_sql = sql.SQL(
        "INSERT INTO {} (document, metadata, embedding) VALUES %s"
    ).format(sql.Identifier(table_name))

    with conn.cursor() as cursor:
        execute_values(cursor, insert_sql, data_to_insert)

# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def run_scraper_etl():
    """Executa scraping + ETL com sistema de checkpoint"""
    conn = None
    model = None

    try:
        # Inicializar checkpoint
        checkpoint = CheckpointManager(CHECKPOINT_FILE) if CHECKPOINT_ENABLED else None

        # Testar conexão
        print("\n" + "=" * 80)
        print("🗄️  CONFIGURANDO BANCO DE DADOS")
        print("=" * 80)

        if not test_db_connection(DB_SETTINGS):
            raise Exception("Não foi possível conectar ao banco")

        conn = psycopg2.connect(**DB_SETTINGS)

        # Criar tabela
        print(f"\nCriando tabela '{DB_TABLE}'...")
        create_pg_table(conn, DB_TABLE, TARGET_MODEL['dimension'])
        conn.commit()

        with conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT COUNT(*) FROM {};").format(sql.Identifier(DB_TABLE)))
            existing_count = cur.fetchone()[0]
            print(f"✓ Tabela pronta")
            print(f"✓ Registros existentes: {existing_count:,}")

        # Carregar modelo
        print(f"\nCarregando modelo {TARGET_MODEL['name']}...")
        model = SentenceTransformer(TARGET_MODEL['name'], device=DEVICE)
        print(f"✅ Modelo carregado em {DEVICE.upper()}")

        # Iniciar scraping
        print("\n" + "=" * 80)
        print("🌐 INICIANDO WEB SCRAPING DO CNJ")
        print("=" * 80)

        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        start_page = checkpoint.get_start_page() if checkpoint else 1
        page_number = start_page

        print(f"\n✓ Iniciando da página: {page_number}")
        print(f"✓ Limite de páginas: {'Sem limite' if PAGE_LIMIT == 0 else PAGE_LIMIT}")
        print(f"✓ Limite de links: {'Sem limite' if LINK_LIMIT == 0 else LINK_LIMIT}")

        all_chunks = []
        total_chunks_inserted = checkpoint.state.get('total_chunks', 0) if checkpoint else 0
        start_time = time.time()

        while True:
            if PAGE_LIMIT > 0 and page_number > PAGE_LIMIT:
                print(f"\n✅ Limite de {PAGE_LIMIT} páginas atingido")
                break

            if LINK_LIMIT > 0 and checkpoint and checkpoint.state['total_links'] >= LINK_LIMIT:
                print(f"\n✅ Limite de {LINK_LIMIT} links atingido")
                break

            current_page_url = LISTING_URL.format(page_number)
            print(f"\n📄 Página {page_number}: {current_page_url}")

            try:
                list_page = session.get(current_page_url, timeout=20).text
            except Exception as e:
                print(f"   ❌ Erro ao buscar página: {e}")
                page_number += 1
                continue

            soup = BeautifulSoup(list_page, 'html.parser')
            table_body = soup.find('table', class_='table')
            table_body = table_body.find('tbody') if table_body else None

            if not table_body or not table_body.find_all('tr'):
                print("   ✅ Fim da paginação")
                break

            # Extrai links
            links_to_process = []
            for row in table_body.find_all('tr'):
                if LINK_LIMIT > 0 and checkpoint and checkpoint.state['total_links'] >= LINK_LIMIT:
                    break
                link_tag = row.find('a', href=True)
                if link_tag:
                    full_url = urljoin(ROOT_URL, link_tag['href'])
                    if not checkpoint or not checkpoint.is_processed(full_url):
                        links_to_process.append(full_url)

            if not links_to_process:
                print("   ℹ️  Todos os links já processados")
                page_number += 1
                continue

            print(f"   Processando {len(links_to_process)} links...")

            # Processa links
            page_chunks = []
            chunks_per_doc = []

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_url = {
                    executor.submit(process_link, url, session, checkpoint): url
                    for url in links_to_process
                }

                for future in tqdm(as_completed(future_to_url),
                                  total=len(links_to_process),
                                  desc="   Links", leave=False):
                    chunks = future.result()
                    if chunks:
                        page_chunks.extend(chunks)
                        chunks_per_doc.append(len(chunks))

            all_chunks.extend(page_chunks)

            if checkpoint:
                checkpoint.update(page=page_number, links=len(links_to_process))

            # Estatísticas
            if chunks_per_doc:
                avg_chunks = sum(chunks_per_doc) / len(chunks_per_doc)
                print(f"   ✓ {len(page_chunks)} chunks gerados de {len(chunks_per_doc)} documentos")
                print(f"     Média: {avg_chunks:.1f} chunks/documento | Min: {min(chunks_per_doc)} | Max: {max(chunks_per_doc)}")
            else:
                print(f"   ⚠️  Nenhum chunk gerado dos links desta página")

            # Processa mega-lotes
            if len(all_chunks) >= PROCESSING_MEGA_BATCH_SIZE or \
               (PAGE_LIMIT > 0 and page_number >= PAGE_LIMIT):

                if all_chunks:
                    print(f"\n--- Processando mega-lote de {len(all_chunks)} chunks ---")

                    # Gera embeddings
                    texts = [chunk['chunk_text'] for chunk in all_chunks]
                    embeddings = model.encode(
                        texts,
                        show_progress_bar=True,
                        batch_size=ENCODE_BATCH_SIZE,
                        normalize_embeddings=True
                    )

                    del texts
                    gc.collect()

                    # Prepara dados
                    data_for_db = []
                    for i, chunk in enumerate(all_chunks):
                        metadata = chunk.get('metadata', {})
                        metadata.update({
                            "document_id": chunk.get("document_id"),
                            "autor": chunk.get("autor"),
                            "tipo_chunk": chunk.get("tipo"),
                            "artigo_pai": chunk.get("artigo_pai")
                        })

                        data_for_db.append((
                            chunk['chunk_text'],
                            json.dumps(metadata, ensure_ascii=False),
                            embeddings[i].tolist()
                        ))

                    del embeddings
                    del all_chunks
                    gc.collect()

                    # Insere no banco
                    print(f"Inserindo {len(data_for_db)} registros...")
                    try:
                        for i in tqdm(range(0, len(data_for_db), DB_BATCH_SIZE),
                                     desc="Inserindo", leave=False):
                            batch = data_for_db[i:i + DB_BATCH_SIZE]
                            batch_insert_data(conn, DB_TABLE, batch)

                        conn.commit()
                        total_chunks_inserted += len(data_for_db)

                        if checkpoint:
                            checkpoint.update(chunks=len(data_for_db))

                        print(f"✅ {len(data_for_db)} registros inseridos")

                    except Exception as e:
                        print(f"❌ Erro na inserção: {e}")
                        conn.rollback()
                        raise
                    finally:
                        del data_for_db
                        gc.collect()
                        if DEVICE == 'cuda':
                            torch.cuda.empty_cache()

                    all_chunks = []

            page_number += 1

        elapsed_time = time.time() - start_time

        # Relatório final
        print("\n" + "=" * 80)
        print("🎉 SCRAPING E ETL CONCLUÍDOS!")
        print("=" * 80)

        if checkpoint:
            print(f"\n📊 ESTATÍSTICAS:")
            print(f"  • Total de páginas: {checkpoint.state['last_page']}")
            print(f"  • Total de links: {checkpoint.state['total_links']}")
            print(f"  • Links processados: {len(checkpoint.state['processed_urls'])}")
            print(f"  • Links com falha: {len(checkpoint.state['failed_urls'])}")
            print(f"  • Total de chunks: {checkpoint.state['total_chunks']:,}")
            print(f"  • Tempo total: {elapsed_time/60:.1f} minutos")
            print(f"  • Velocidade: {checkpoint.state['total_chunks']/(elapsed_time/60):.1f} chunks/min")

        with conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT COUNT(*) FROM {};").format(sql.Identifier(DB_TABLE)))
            final_count = cur.fetchone()[0]

        print(f"\n💾 BANCO DE DADOS:")
        print(f"  • Tabela: {DB_SCHEMA}.{DB_TABLE}")
        print(f"  • Total de registros: {final_count:,}")
        print(f"  • Novos registros: {final_count - existing_count:,}")

        # Limpa checkpoint se concluído
        if checkpoint and CHECKPOINT_FILE.exists():
            print(f"\n🧹 Limpando checkpoint...")
            CHECKPOINT_FILE.unlink()
            print("✓ Checkpoint removido (processo concluído)")

    except KeyboardInterrupt:
        print(f"\n\n⚠️  PROCESSO INTERROMPIDO PELO USUÁRIO")
        print(f"\n💾 Checkpoint salvo! Execute novamente para continuar de onde parou.")
        if conn:
            conn.rollback()

    except Exception as e:
        print(f"\n{'=' * 80}")
        print(f"❌ ERRO NO PIPELINE")
        print(f"{'=' * 80}")
        print(f"{e}\n")
        import traceback
        traceback.print_exc()
        print(f"\n💾 Checkpoint salvo! Execute novamente para continuar.")
        if conn:
            conn.rollback()

    finally:
        print("\n" + "=" * 80)
        print("🧹 LIMPANDO RECURSOS")
        print("=" * 80)

        if conn:
            conn.close()
            print("✓ Conexão fechada")
        if model:
            del model
            print("✓ Modelo removido")

        gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        print("✓ Memória limpa")

        print("\n" + "=" * 80)
        print("✅ PROCESSO FINALIZADO")
        print("=" * 80)

# ==============================================================================
# EXECUÇÃO
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 INICIANDO ETL CNJ")
    print("=" * 80)
    
    # Validação das configurações
    print("\n⚠️  VERIFICANDO CONFIGURAÇÕES...")
    
    if DB_SETTINGS["password"] == "sua_senha_aqui":
        print("\n❌ ERRO: Você precisa configurar a senha do banco de dados!")
        print("   Edite a variável DB_SETTINGS no início do arquivo.")
        exit(1)
    
    print("✓ Configurações validadas")
    
    # Instala dependências se necessário
    try:
        import langchain
        import sentence_transformers
    except ImportError:
        print("\n⚠️  Instalando dependências faltantes...")
        import subprocess
        subprocess.run([
            "pip", "install", "-q",
            "beautifulsoup4", "psycopg2-binary", "tqdm", 
            "sentence-transformers", "langchain", "requests", "torch"
        ])
        print("✓ Dependências instaladas")
    
    # Inicia o processo
    try:
        run_scraper_etl()
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")
        import traceback
        traceback.print_exc()