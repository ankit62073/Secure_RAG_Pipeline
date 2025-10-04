"""
Enhanced Semantic enrichment with hierarchical docstring generation:
• generates docstrings with GPT for functions, files, folders, and branches
• builds embeddings with sentence-transformers for all levels
• handles large files through intelligent chunking and summarization
• creates layered summaries (function → file → folder → branch)
• writes results back to Neo4j

Key Changes:
1. Added hierarchical processing methods
2. Implemented smart chunking for large files
3. Added file size analysis and adaptive processing
4. Created aggregation methods for folder and branch summaries
5. Enhanced the main processing pipeline
"""

from __future__ import annotations

import os, re, json, time, logging, asyncio
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv

# ─── Global Neo4j utilities ────────────────────────────────────
from app.configs.neo4j_setup import get_driver                   # singleton pool
from app.ai.utils.neo4j_utils import retry_on_connection_error   # auto-refresh

load_dotenv()
logger = logging.getLogger(__name__)

# ─── Enhanced data structures ──────────────────────────────────
@dataclass
class DocstringRequest:
    node_id: str
    text: str
    node_type: str  # NEW: 'function', 'file', 'folder', 'branch'
    context: Dict = None  # NEW: Additional context for hierarchical processing

@dataclass
class DocstringData:
    node_id: str
    docstring: str
    tags: List[str]
    node_type: str  # NEW: Track node type
    summary: str = ""  # NEW: Brief summary for aggregation
    complexity_score: int = 0  # NEW: Complexity indicator

@dataclass
class DocstringResponse:
    docstrings: List[DocstringData]

@dataclass
class FileAnalysis:
    """NEW: Structure to analyze file characteristics"""
    size_category: str  # 'small', 'medium', 'large', 'huge'
    line_count: int
    char_count: int
    chunk_strategy: str  # 'direct', 'semantic_chunks', 'hierarchical_summary'
    estimated_tokens: int

# ─── Enhanced Main class ───────────────────────────────────────
class HierarchicalSemanticEnrichment:
    """
    Enhanced semantic enrichment supporting hierarchical docstring generation
    for functions, files, folders, and branches with intelligent size handling.
    """

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        neo_db: str | None = None,
        per_node_limit: int | None = None,
    ) -> None:
        try:
            # Embedding + OpenAI
            self.embedding_model   = SentenceTransformer("all-mpnet-base-v2")
            self.gpt_model         = os.getenv("OPENAI_MODEL")
            self.openai_client     = openai.AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )

            self.parallel_requests = int(os.getenv("PARALLEL_REQUESTS", "10"))
            self.per_node_limit    = per_node_limit or int(os.getenv("PER_NODE_LIMIT", "10"))

            # NEW: File size thresholds (in characters)
            self.size_thresholds = {
                'small': 2000000,      # < 2K chars - direct processing
                'medium': 4000000,     # 2K-8K chars - semantic chunking
                'large': 8000000,     # 8K-32K chars - hierarchical summary
                'huge': 10000000      # > 32K chars - multi-level chunking
            }
            
            # NEW: Token estimation (rough: 1 token ≈ 4 characters)
            self.max_context_tokens = int(os.getenv("MAX_CONTEXT_TOKENS", "16000"))

            # Neo4j
            self.driver           = get_driver()
            self.neo4j_database   = neo_db or os.getenv("NEO4J_DEFAULT_DB", "neo4j")

            if not self.verify_connection():
                raise ConnectionError("Failed to connect to Neo4j")

            self._setup_vector_indexes()

        except Exception as exc:
            logger.error(f"[HierarchicalSemanticEnrich] init failed: {exc}")
            raise

    # ── Connectivity helpers ──────────────────────────────────
    def verify_connection(self) -> bool:
        try:
            with self.driver.session(database=self.neo4j_database) as s:
                return s.run("RETURN 1 as ok").single().get("ok") == 1
        except Exception as exc:
            logger.error(f"[HierarchicalSemanticEnrich] connectivity check failed: {exc}")
            return False

    @retry_on_connection_error()
    def _setup_vector_indexes(self) -> None:
        # ENHANCED: Added indexes for all node types
        vec_labels = [
            ("class_vector_index",     "CLASS"),
            ("function_vector_index",  "FUNCTION"),
            ("method_vector_index",    "METHOD"),
            ("interface_vector_index", "INTERFACE"),
            ("file_vector_index",      "FILE"),        # NEW
            ("folder_vector_index",    "FOLDER"),      # NEW
            ("branch_vector_index",    "BRANCH"),      # NEW
        ]
        full_text_stmt = """
        CREATE FULLTEXT INDEX code_fulltext_index IF NOT EXISTS
        FOR (n:CLASS|FUNCTION|METHOD|INTERFACE|FILE|FOLDER|BRANCH)
        ON EACH [n.name, n.content, n.text, n.docstring, n.summary]
        """

        with self.driver.session(database=self.neo4j_database) as s:
            existing = {r["name"]: r["type"] for r in s.run("SHOW INDEXES YIELD name, type")}
            for name, label in vec_labels:
                if name not in existing:
                    logger.info(f"[Neo4j] creating vector index {name}")
                    s.run(f"""
                        CREATE VECTOR INDEX {name} IF NOT EXISTS
                        FOR (n:{label}) ON (n.embedding)
                        OPTIONS {{
                          indexConfig: {{
                            `vector.dimensions`: 768,
                            `vector.similarity_function`: 'cosine'
                          }}
                        }}""")
            if "code_fulltext_index" not in existing:
                logger.info("[Neo4j] creating full-text index code_fulltext_index")
                s.run(full_text_stmt)

    # ── NEW: File Analysis Methods ─────────────────────────────
    def analyze_file_characteristics(self, content: str, file_path: str = "") -> FileAnalysis:
        """Analyze file to determine processing strategy based on size and complexity."""
        char_count = len(content)
        line_count = content.count('\n') + 1
        estimated_tokens = char_count // 4  # Rough estimation
        
        # Determine size category
        if char_count < self.size_thresholds['small']:
            size_category = 'small'
            chunk_strategy = 'direct'
        elif char_count < self.size_thresholds['medium']:
            size_category = 'medium'
            chunk_strategy = 'semantic_chunks'
        elif char_count < self.size_thresholds['large']:
            size_category = 'large'
            chunk_strategy = 'hierarchical_summary'
        else:
            size_category = 'huge'
            chunk_strategy = 'multi_level_chunking'
        
        logger.debug(f"File analysis: {file_path} - {size_category} ({char_count} chars, ~{estimated_tokens} tokens)")
        
        return FileAnalysis(
            size_category=size_category,
            line_count=line_count,
            char_count=char_count,
            chunk_strategy=chunk_strategy,
            estimated_tokens=estimated_tokens
        )

    def create_semantic_chunks(self, content: str, max_chunk_size: int = 6000) -> List[str]:
        """NEW: Create semantically meaningful chunks for large files."""
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        # Try to break at logical boundaries (classes, functions, etc.)
        logical_boundaries = ['class ', 'def ', 'function ', 'interface ', '# ===', '# ---']
        
        for line in lines:
            line_with_newline = line + '\n'
            line_size = len(line_with_newline)
            
            # Check if we should start a new chunk
            should_break = (
                current_size + line_size > max_chunk_size and 
                current_chunk and
                any(line.strip().startswith(boundary) for boundary in logical_boundaries)
            )
            
            if should_break:
                chunks.append(''.join(current_chunk).strip())
                current_chunk = [line_with_newline]
                current_size = line_size
            else:
                current_chunk.append(line_with_newline)
                current_size += line_size
        
        # Add the last chunk
        if current_chunk:
            chunks.append(''.join(current_chunk).strip())
        
        logger.debug(f"Created {len(chunks)} semantic chunks")
        return chunks

    async def process_large_file_hierarchically(self, content: str, file_path: str) -> Dict[str, str]:
        """NEW: Process large files using hierarchical summarization."""
        chunks = self.create_semantic_chunks(content)
        
        # Step 1: Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            summary = await self.generate_chunk_summary(chunk, f"chunk_{i+1}_of_{len(chunks)}")
            chunk_summaries.append(summary)
        
        # Step 2: Create overall file summary from chunk summaries
        combined_summary = "\n".join([f"Section {i+1}: {summary}" for i, summary in enumerate(chunk_summaries)])
        
        # Step 3: Generate final docstring based on summaries
        file_docstring = await self.generate_file_docstring_from_summaries(
            combined_summary, file_path, len(chunks)
        )
        
        return {
            'docstring': file_docstring,
            'summary': combined_summary[:500] + "..." if len(combined_summary) > 500 else combined_summary,
            'chunk_count': len(chunks)
        }

    async def generate_chunk_summary(self, chunk: str, chunk_id: str) -> str:
        """NEW: Generate a summary for a code chunk."""
        prompt = f"""
        Analyze this code chunk and provide a concise summary (2-3 sentences) of its main functionality:

        ```
        {chunk[:3000]}  # Truncate very large chunks
        ```

        Focus on:
        - Main classes/functions defined
        - Primary purpose and functionality
        - Key algorithms or patterns used
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.gpt_model,
                messages=[
                    {"role": "system", "content": "You are a code analysis expert. Provide concise, technical summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating chunk summary for {chunk_id}: {e}")
            return f"Code chunk containing various functions and logic (analysis failed)"

    async def generate_file_docstring_from_summaries(self, summaries: str, file_path: str, chunk_count: int) -> str:
        """NEW: Generate file-level docstring from chunk summaries."""
        file_name = Path(file_path).name if file_path else "file"
        
        prompt = f"""
        Based on the following section summaries from a code file "{file_name}" ({chunk_count} sections), 
        create a comprehensive file-level docstring:

        {summaries}

        Create a docstring that:
        1. Starts with a clear one-line summary of the file's purpose
        2. Describes the main components and their interactions
        3. Mentions key functionality and patterns
        4. Is suitable for a senior developer to quickly understand the file's role

        Keep it concise but informative (4-6 sentences).
        Return ONLY the docstring text, no JSON formatting.
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.gpt_model,
                messages=[
                    {"role": "system", "content": "You are a senior software architect creating file-level documentation. Return only the docstring text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating file docstring from summaries: {e}")
            return f"Module containing {chunk_count} sections with various functionality (analysis incomplete)"

    # ── PUBLIC HELPERS (enhanced for hierarchy) ────────────────
    def log_graph_stats(self, repo_id: str) -> None:
        # ENHANCED: Include all node types in stats
        q = """
        MATCH (n)
        WHERE n.repoId = $id OR n.repo_id = $id
        OPTIONAL MATCH (n)-[r]-()
        WITH n, r
        RETURN 
            COUNT(DISTINCT n) AS total_nodes, 
            COUNT(DISTINCT r) AS total_rels,
            COUNT(DISTINCT CASE WHEN n:FUNCTION THEN n END) as functions,
            COUNT(DISTINCT CASE WHEN n:FILE THEN n END) as files,
            COUNT(DISTINCT CASE WHEN n:FOLDER THEN n END) as folders,
            COUNT(DISTINCT CASE WHEN n:BRANCH THEN n END) as branches
        """
        try:
            with self.driver.session(database=self.neo4j_database) as s:
                rec = s.run(q, id=repo_id).single()
                logger.info(f"[Neo4j] repo {repo_id}: {rec['total_nodes']} total nodes ({rec['functions']} functions, {rec['files']} files, {rec['folders']} folders, {rec['branches']} branches), {rec['total_rels']} relationships")
        except Exception as exc:
            logger.error(f"[Neo4j] stats error: {exc}")

    async def generate_embedding(self, text: str) -> List[float]:
        text = text.strip()
        if not text:
            return []
        # Truncate very long text to prevent embedding issues
        if len(text) > 8000:
            text = text[:8000] + "..."
        loop = asyncio.get_event_loop()
        emb  = await loop.run_in_executor(
            None, lambda: self.embedding_model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        )
        return emb.tolist()

    async def generate_tags(self, text: str, node_type: str = "function") -> List[str]:
        # ENHANCED: Context-aware tag generation based on node type
        tag_instructions = {
            'function': "Focus on function-specific tags: UTILITY, API, DATABASE, AUTH, etc.",
            'file': "Focus on file-level tags: MODULE, CONFIGURATION, INTEGRATION, CORE_LOGIC, etc.",
            'folder': "Focus on architectural tags: SERVICE, COMPONENT, LAYER, DOMAIN, etc.",
            'branch': "Focus on high-level tags: FEATURE, RELEASE, EXPERIMENTAL, HOTFIX, etc."
        }
        
        instruction = tag_instructions.get(node_type, tag_instructions['function'])
        
        delay, retries = 1, 5
        for attempt in range(retries):
            try:
                resp = await self.openai_client.chat.completions.create(
                    model=self.gpt_model,
                    messages=[
                        {"role": "system",
                         "content": f"You are a code analysis assistant for {node_type}-level tagging. "
                                    f"{instruction} Return a comma-separated list of 2-5 relevant tags."},
                        {"role": "user", "content": text[:2000]}  # Truncate for tag generation
                    ],
                    max_tokens=100, temperature=0.3,
                )
                return [t.strip() for t in resp.choices[0].message.content.split(",")]
            except openai.RateLimitError:
                logger.info(f"Rate-limit; retry {attempt+1}/{retries}")
                await asyncio.sleep(delay)
                delay *= 2
            except Exception as exc:
                logger.error(f"Tag generation error: {exc}")
                return []
        raise RuntimeError("Max retries exceeded for tag generation")

    # ── ENHANCED: Hierarchical Processing Methods ──────────────
    async def process_node_batch_hierarchical(self, batch: List[Dict]) -> List[DocstringData]:
        """ENHANCED: Process nodes with hierarchy-aware logic."""
        semaphore = asyncio.Semaphore(self.per_node_limit or 10)
        results = []
        successful = 0
        failed = 0

        async def process_node(node):
            nonlocal successful, failed
            
            async with semaphore:
                try:
                    node_id = node.get('node_id')
                    node_type = self.determine_node_type(node)  # NEW: Determine type
                    
                    if not node_id:
                        logger.warning(f"Skipping node without ID")
                        failed += 1
                        return None
                    
                    logger.debug(f"Processing {node_type} node: {node_id}")
                    
                    # Check for existing docstring
                    existing_docstring = node.get('existing_docstring')
                    if existing_docstring:
                        content = node.get('content', '') or node.get('text', '')
                        text = f"{existing_docstring}\n{content}"
                        tags = await self.generate_tags(text, node_type)
                        successful += 1
                        return DocstringData(
                            node_id=node_id,
                            docstring=existing_docstring,
                            tags=tags or [],
                            node_type=node_type,
                            summary=existing_docstring[:200] + "..." if len(existing_docstring) > 200 else existing_docstring
                        )
                    else:
                        # Generate new docstring with type-specific logic
                        docstring_data = await self.generate_hierarchical_docstring(node, node_type)
                        if docstring_data:
                            successful += 1
                            return docstring_data
                        else:
                            failed += 1
                            return None
                            
                except Exception as e:
                    logger.error(f"Error processing node: {str(e)}")
                    failed += 1
                    return None

        # Process all nodes in parallel
        start_time = asyncio.get_event_loop().time()
        tasks = [process_node(node) for node in batch]
        node_results = await asyncio.gather(*tasks, return_exceptions=False)
        
        results = [r for r in node_results if r is not None]
        
        elapsed = asyncio.get_event_loop().time() - start_time
        logger.info(f"Processed {len(batch)} hierarchical nodes in {elapsed:.2f}s: {successful} successful, {failed} failed")
        
        return results

    def determine_node_type(self, node: Dict) -> str:
        """NEW: Determine node type from node data."""
        # Check node labels or properties to determine type
        labels = node.get('labels', [])
        if isinstance(labels, str):
            labels = [labels]
        
        for label in ['FUNCTION', 'METHOD', 'CLASS', 'INTERFACE', 'FILE', 'FOLDER', 'BRANCH']:
            if label in labels:
                return label.lower()
        
        # Fallback: try to infer from node properties
        if 'file_path' in node or 'path' in node:
            return 'file'
        elif 'folder_path' in node or 'directory' in node:
            return 'folder'
        elif 'branch_name' in node:
            return 'branch'
        else:
            return 'function'  # Default fallback

    async def generate_hierarchical_docstring(self, node: Dict, node_type: str) -> Optional[DocstringData]:
        """NEW: Generate docstring using type-specific strategies."""
        try:
            content = node.get('content', '') or node.get('text', '')
            
            # Log content availability for debugging
            logger.debug(f"Processing {node_type} node {node.get('node_id')} with content length: {len(content) if content else 0}")
            
            if not content or content.strip() == '':
                logger.warning(f"No content available for {node_type} node {node.get('node_id')}")
                return None

            if node_type == 'file':
                return await self.process_file_node(node, content)
            elif node_type == 'folder':
                return await self.process_folder_node(node, content)
            elif node_type == 'branch':
                return await self.process_branch_node(node, content)
            else:
                # Function, method, class, interface - use original logic
                return await self.process_function_node(node, content)
                
        except Exception as e:
            logger.error(f"Error in hierarchical docstring generation for {node_type}: {e}")
            return None

    async def process_file_node(self, node: Dict, content: str) -> DocstringData:
        """NEW: Process file-level nodes with size-aware strategies."""
        node_id = node['node_id']
        file_path = node.get('file_path', node.get('path', ''))
        
        # Analyze file characteristics
        analysis = self.analyze_file_characteristics(content, file_path)
        
        if analysis.chunk_strategy == 'direct':
            # Small file - direct processing
            response = await self.generate_response([DocstringRequest(
                node_id=node_id,
                text=content,
                node_type='file'
            )])
            result = response.docstrings[0] if response.docstrings else None
        elif analysis.chunk_strategy in ['semantic_chunks', 'hierarchical_summary', 'multi_level_chunking']:
            # Large file - hierarchical processing
            file_result = await self.process_large_file_hierarchically(content, file_path)
            tags = await self.generate_tags(file_result['summary'], 'file')
            result = DocstringData(
                node_id=node_id,
                docstring=file_result['docstring'],
                tags=tags,
                node_type='file',
                summary=file_result['summary'],
                complexity_score=file_result.get('chunk_count', 1)
            )
        
        if result:
            result.node_type = 'file'
            logger.debug(f"Generated file docstring for {file_path} using {analysis.chunk_strategy} strategy")
        
        return result

    async def process_folder_node(self, node: Dict, content: str) -> DocstringData:
        """NEW: Process folder-level nodes by aggregating file summaries (UPDATED)."""
        node_id = node['node_id']
        folder_path = node.get('folder_path', node.get('path', ''))
        folder_name = node.get('name', Path(folder_path).name if folder_path else 'Unknown Folder')
        
        logger.debug(f"Processing folder node: {node_id} at path: {folder_path}")
        
        # Get summaries of files in this folder from the graph
        file_summaries = await self.get_folder_file_summaries(node_id, folder_path)
        
        if not file_summaries:
            logger.warning(f"No file summaries found for folder {node_id}")
            # Create a basic docstring based on folder name and path
            basic_docstring = f"Folder '{folder_name}' containing code files and components."
            tags = await self.generate_tags(f"Folder {folder_name} {folder_path}", 'folder')
            return DocstringData(
                node_id=node_id,
                docstring=basic_docstring,
                tags=tags or ['COMPONENT'],
                node_type='folder',
                summary=basic_docstring,
                complexity_score=0
            )
        
        # Generate folder-level docstring from file summaries
        folder_context = f"Folder: {folder_name} (Path: {folder_path})\nContains {len(file_summaries)} files:\n"
        folder_context += "\n".join([f"- {summary}" for summary in file_summaries[:10]])  # Limit context
        
        logger.debug(f"Generated context for folder {node_id}: {len(folder_context)} characters")
        
        try:
            response = await self.generate_response([DocstringRequest(
                node_id=node_id,
                text=folder_context,
                node_type='folder'
            )])
            
            result = response.docstrings[0] if response.docstrings else None
            if result:
                result.node_type = 'folder'
                result.complexity_score = len(file_summaries)
                logger.debug(f"Generated folder docstring for {folder_name}")
                return result
            else:
                logger.error(f"No docstring generated for folder {node_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing folder node {node_id}: {e}")
            return None

    async def process_branch_node(self, node: Dict, content: str) -> DocstringData:
        """NEW: Process branch-level nodes by aggregating folder summaries."""
        node_id = node['node_id']
        branch_name = node.get('branch_name', node.get('name', ''))
        
        # Get summaries of folders/major components in this branch
        component_summaries = await self.get_branch_component_summaries(node_id)
        
        # Generate branch-level docstring
        branch_context = f"Branch: {branch_name}\nContains {len(component_summaries)} major components\n"
        branch_context += "\n".join([f"- {summary}" for summary in component_summaries[:15]])  # Limit context
        
        response = await self.generate_response([DocstringRequest(
            node_id=node_id,
            text=branch_context,
            node_type='branch'
        )])
        
        result = response.docstrings[0] if response.docstrings else None
        if result:
            result.node_type = 'branch'
            result.complexity_score = len(component_summaries)
        
        return result

    async def process_function_node(self, node: Dict, content: str) -> DocstringData:
        """Process function/method/class nodes (fixed logic)."""
        node_id = node['node_id']
        
        # Check if content exists
        if not content or content.strip() == '':
            logger.warning(f"No content found for function node {node_id}")
            return None
        
        try:
            response = await self.generate_response([DocstringRequest(
                node_id=node_id,
                text=content,
                node_type='function'
            )])
            
            if response and response.docstrings:
                result = response.docstrings[0]
                result.node_type = 'function'
                return result
            else:
                logger.error(f"No docstring generated for function node {node_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing function node {node_id}: {e}")
            return None

    async def get_folder_file_summaries(self, folder_node_id: str, folder_path: str = "") -> List[str]:
        """UPDATED: Get summaries of files within a folder using multiple strategies."""
        try:
            with self.driver.session(database=self.neo4j_database) as session:
                summaries = []
                
                # Strategy 1: Try to find files directly connected to this folder
                result1 = session.run("""
                    MATCH (folder)-[:CONTAINS]->(file:FILE)
                    WHERE folder.node_id = $folder_id
                    AND (file.summary IS NOT NULL OR file.docstring IS NOT NULL)
                    RETURN COALESCE(file.summary, file.docstring, 'File: ' + COALESCE(file.name, 'unnamed')) as summary
                    LIMIT 15
                """, {"folder_id": folder_node_id})
                
                for record in result1:
                    if record["summary"]:
                        summaries.append(record["summary"])
                
                # Strategy 2: If no direct connections, try path-based matching
                if not summaries and folder_path:
                    # Extract the folder path pattern for matching
                    path_pattern = folder_path.replace('\\', '/').rstrip('/')
                    logger.debug(f"Trying path-based matching for folder: {path_pattern}")
                    
                    result2 = session.run("""
                        MATCH (file:FILE)
                        WHERE file.path IS NOT NULL 
                        AND (file.path CONTAINS $path_pattern OR file.path STARTS WITH $path_with_slash)
                        AND (file.summary IS NOT NULL OR file.docstring IS NOT NULL)
                        RETURN COALESCE(file.summary, file.docstring, 'File: ' + COALESCE(file.name, file.path)) as summary,
                            file.path as file_path
                        LIMIT 15
                    """, {
                        "path_pattern": path_pattern,
                        "path_with_slash": path_pattern + "/"
                    })
                    
                    for record in result2:
                        if record["summary"]:
                            summaries.append(f"File ({Path(record['file_path']).name}): {record['summary']}")
                
                # Strategy 3: If still no summaries, look for files by node_id pattern
                if not summaries:
                    # Extract repository and path info from folder node_id
                    # Format: "notp-backend_main::folder_controllers\helper_functions"
                    if "::" in folder_node_id:
                        repo_part, folder_part = folder_node_id.split("::", 1)
                        folder_part = folder_part.replace("folder_", "")  # Remove folder_ prefix
                        
                        result3 = session.run("""
                            MATCH (file:FILE)
                            WHERE file.node_id CONTAINS $repo_part
                            AND file.node_id CONTAINS $folder_part
                            AND (file.summary IS NOT NULL OR file.docstring IS NOT NULL)
                            RETURN COALESCE(file.summary, file.docstring, 'File: ' + COALESCE(file.name, 'unnamed')) as summary
                            LIMIT 15
                        """, {
                            "repo_part": repo_part,
                            "folder_part": folder_part
                        })
                        
                        for record in result3:
                            if record["summary"]:
                                summaries.append(record["summary"])
                
                logger.debug(f"Found {len(summaries)} file summaries for folder {folder_node_id}")
                return summaries
                
        except Exception as e:
            logger.error(f"Error getting folder file summaries for {folder_node_id}: {e}")
            return []

    async def get_branch_component_summaries(self, branch_node_id: str) -> List[str]:
        """NEW: Get summaries of major components within a branch."""
        try:
            with self.driver.session(database=self.neo4j_database) as session:
                # Get summaries from folders and major files
                result = session.run("""
                    MATCH (branch)-[:CONTAINS*1..2]->(component)
                    WHERE branch.node_id = $branch_id
                    AND (component:FOLDER OR component:FILE)
                    AND component.summary IS NOT NULL
                    RETURN component.summary as summary, component.complexity_score as complexity
                    ORDER BY complexity DESC
                    LIMIT 25
                """, {"branch_id": branch_node_id})
                return [record["summary"] for record in result]
        except Exception as e:
            logger.error(f"Error getting branch component summaries: {e}")
            return []

    # ── ENHANCED: Main Processing Pipeline ─────────────────────
    async def generate_docstrings_hierarchical(self, repo_id: str) -> Dict[str, DocstringResponse]:
        """ENHANCED: Main method with hierarchical processing pipeline and concurrent processing for independent node types."""
        if not self.driver:
            raise ValueError("Neo4j connection not initialized")

        try:
            self.log_graph_stats(repo_id)
            
            if not self.verify_connection():
                logger.error("Neo4j connection failed verification")
                raise ConnectionError("Neo4j connection is not active")
            
            # **CHANGE 1: Group node types into concurrent and sequential categories**
            # These can run concurrently as they are independent
            independent_types = ['function', 'method', 'class', 'interface', 'file']
            # These need to run after independent types are complete
            dependent_types = ['folder', 'branch']
            
            all_docstrings = {"docstrings": []}
            
            # **CHANGE 2: Process independent node types concurrently**
            logger.info("Starting concurrent processing of independent node types...")
            concurrent_tasks = []
            
            for node_type in independent_types:
                task = self._process_node_type_async(node_type)
                concurrent_tasks.append((node_type, task))
            
            # Execute all independent tasks concurrently
            concurrent_results = await asyncio.gather(
                *[task for _, task in concurrent_tasks], 
                return_exceptions=True
            )
            
            # Process results from concurrent execution
            for i, (node_type, result) in enumerate(zip([nt for nt, _ in concurrent_tasks], concurrent_results)):
                if isinstance(result, Exception):
                    logger.error(f"Error processing {node_type} nodes concurrently: {result}")
                    continue
                
                if result:
                    all_docstrings["docstrings"].extend(result)
                    logger.info(f"Concurrent processing of {node_type} completed: {len(result)} nodes processed")
            
            logger.info(f"Concurrent processing phase completed. Total processed: {len(all_docstrings['docstrings'])} nodes")
            
            # **CHANGE 3: Process dependent types sequentially after concurrent phase**
            logger.info("Starting sequential processing of dependent node types...")
            for node_type in dependent_types:
                logger.info(f"Processing {node_type} level nodes...")
                
                try:
                    result = await self._process_node_type_async(node_type)
                    if result:
                        all_docstrings["docstrings"].extend(result)
                        logger.info(f"Sequential processing of {node_type} completed: {len(result)} nodes processed")
                except Exception as e:
                    logger.error(f"Error processing {node_type} nodes sequentially: {e}")
                    continue
            
            total_processed = len(all_docstrings["docstrings"])
            logger.info(f"Hierarchical enrichment complete: {total_processed} nodes processed across all levels")
            
            return all_docstrings

        except Exception as e:
            logger.error(f"Fatal hierarchical enrichment error: {e}")
            raise

    async def _process_node_type_async(self, node_type: str) -> List[DocstringData]:
        """
        **NEW HELPER METHOD**: Asynchronously process all nodes of a specific type.
        This method encapsulates the batch processing logic for concurrent execution.
        """
        try:
            nodes = self.get_nodes_for_enrichment_by_type(node_type)
            if not nodes:
                logger.info(f"No {node_type} nodes require enrichment")
                return []
            
            logger.info(f"Found {len(nodes)} {node_type} nodes for enrichment")
            
            all_docstrings = []
            batch_size = self.get_optimal_batch_size(node_type)
            total_batches = (len(nodes) + batch_size - 1) // batch_size
            
            # **CHANGE 4: Process batches for this node type**
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                logger.info(f"Processing {node_type} batch {batch_num}/{total_batches} ({len(batch)} nodes)")
                
                try:
                    docstrings = await self.process_node_batch_hierarchical(batch)
                    
                    if docstrings:
                        # **CHANGE 5: Update graph immediately after each batch**
                        await self.update_nodes_in_graph_hierarchical(
                            self._get_repo_id_from_nodes(batch), 
                            DocstringResponse(docstrings=docstrings)
                        )
                        all_docstrings.extend(docstrings)
                        logger.info(f"Successfully processed {len(docstrings)} {node_type} nodes in batch {batch_num}")
                    
                except Exception as e:
                    logger.error(f"Error processing {node_type} batch {batch_num}: {e}")
                    continue
            
            logger.info(f"Completed processing {node_type}: {len(all_docstrings)} total nodes")
            return all_docstrings
            
        except Exception as e:
            logger.error(f"Error in _process_node_type_async for {node_type}: {e}")
            return []

    def _get_repo_id_from_nodes(self, batch: List[Dict]) -> str:
        """
        **NEW HELPER METHOD**: Extract repo_id from node batch for graph updates.
        """
        for node in batch:
            node_id = node.get('node_id', '')
            if '::' in node_id:
                # Extract repo part from node_id format: "repo_branch::path"
                return node_id.split('::')[0]
        
        # Fallback - you might need to adjust this based on your node_id format
        return "unknown_repo"

    def get_optimal_batch_size(self, node_type: str) -> int:
        """NEW: Get optimal batch size based on node type."""
        batch_sizes = {
            'function': 50,
            'method': 50,
            'class': 30,
            'interface': 30,
            'file': 10,      # Files can be large
            'folder': 5,     # Folders require aggregation
            'branch': 2      # Branches are complex
        }
        return batch_sizes.get(node_type, 20)

    def get_nodes_for_enrichment_by_type(self, node_type: str) -> List[Dict]:
        """ENHANCED: Get nodes by type that need enrichment."""
        try:
            if not self.driver:
                raise ValueError("Neo4j connection not initialized")
                
            # Map node type to Neo4j label
            type_label_map = {
                'function': 'FUNCTION',
                'method': 'METHOD', 
                'class': 'CLASS',
                'interface': 'INTERFACE',
                'file': 'FILE',
                'folder': 'FOLDER',
                'branch': 'BRANCH'
            }
            
            label = type_label_map.get(node_type, 'FUNCTION')
            print(f"Querying for nodes of type: {node_type} with label: {label}")
                
            with self.driver.session(database=self.neo4j_database) as session:
                # Enhanced query to get different types of nodes
                if node_type in ['function', 'method', 'class', 'interface']:
                    # Original logic for code elements - REMOVED tags check from WHERE clause
                    query = f"""
                        MATCH (n:{label})
                        WHERE (
                            n.docstring IS NULL OR 
                            n.docstring = '' OR 
                            n.embedding IS NULL
                        )
                        RETURN n.node_id as node_id, 
                            n.content as content,
                            n.docstring as existing_docstring,
                            labels(n) as labels
                    """
                elif node_type == 'file':
                    # File nodes may have different properties - REMOVED tags check from WHERE clause
                    query = f"""
                        MATCH (n:{label})
                        WHERE (
                            n.docstring IS NULL OR 
                            n.docstring = '' OR 
                            n.embedding IS NULL OR 
                            n.summary IS NULL
                        )
                        RETURN n.node_id as node_id, 
                            n.text as content,
                            n.path as file_path,
                            n.docstring as existing_docstring,
                            labels(n) as labels
                    """
                elif node_type == 'folder':
                    # Folder nodes need different handling
                    query = f"""
                        MATCH (n:{label})
                        WHERE (
                            n.docstring IS NULL OR 
                            n.docstring = '' OR 
                            n.embedding IS NULL OR 
                            n.summary IS NULL
                        )
                        RETURN n.node_id as node_id, 
                               n.name as content,
                               n.path as folder_path,
                               n.description as existing_docstring,
                               labels(n) as labels
                    """
                else:
                    return []
                
                result = session.run(query)
                nodes = [dict(record) for record in result]
                
                logger.debug(f"Retrieved {len(nodes)} {node_type} nodes needing enrichment")
                return nodes
                
        except Exception as e:
            logger.error(f"Failed to retrieve {node_type} nodes: {e}")
            return []

    # ── ENHANCED: Response Generation ──────────────────────────
    async def generate_response(self, batch: List[DocstringRequest]) -> DocstringResponse:
        """ENHANCED: Generate responses with type-specific prompts."""
        
        # Group requests by type for optimized processing
        type_groups = {}
        for request in batch:
            node_type = request.node_type
            if node_type not in type_groups:
                type_groups[node_type] = []
            type_groups[node_type].append(request)
        
        all_docstrings = []
        
        # Process each type group with specialized prompts
        for node_type, requests in type_groups.items():
            try:
                if node_type == 'file':
                    prompt = self.get_file_level_prompt(requests)
                elif node_type == 'folder':
                    prompt = self.get_folder_level_prompt(requests)
                # elif node_type == 'branch':
                #     prompt = self.get_branch_level_prompt(requests)
                else:
                    prompt = self.get_function_level_prompt(requests)
                
                response_data = await self.call_openai_with_prompt(prompt, node_type)
                all_docstrings.extend(response_data)
                
            except Exception as e:
                logger.error(f"Error processing {node_type} requests: {e}")
                continue
        
        return DocstringResponse(docstrings=all_docstrings)

    def get_file_level_prompt(self, requests: List[DocstringRequest]) -> str:
        """NEW: Generate file-level specific prompt."""
        code_snippets = ""
        for request in requests:
            # Truncate very large files for the prompt
            content = request.text[:10000] + "..." if len(request.text) > 10000 else request.text
            code_snippets += f"node_id: {request.node_id}\nFile Content:\n```\n{content}\n```\n\n"

        return f"""
        You are a senior software architect analyzing code files. Generate comprehensive file-level docstrings that describe the overall purpose, architecture, and key components of each file.

        For each file, provide:
        1. **Purpose**: Clear summary of what this file does in the system
        2. **Key Components**: Main classes, functions, or modules defined
        3. **Architecture**: How components interact and file's role in larger system
        4. **Dependencies**: Important imports or external dependencies
        5. **Usage Context**: When and how this file would be used

        **File-Level Tags** (choose 2-5):
        - MODULE: Core module/component
        - CONFIGURATION: Config/settings file  
        - INTEGRATION: External service integration
        - CORE_LOGIC: Business logic implementation
        - UTILITY: Utility/helper functions
        - DATA_MODEL: Data structures/models
        - SERVICE_LAYER: Service layer implementation
        - CONTROLLER: Request/response handling
        - MIDDLEWARE: Middleware components
        - TESTING: Test files
        - DOCUMENTATION: Documentation files

        Response format: JSON with "docstrings" array containing objects with node_id, docstring, tags, and summary fields.

        Files to analyze:
        {code_snippets}
        """

    def get_folder_level_prompt(self, requests: List[DocstringRequest]) -> str:
        """UPDATED: Generate folder-level specific prompt with better structure."""
        folder_info = ""
        for request in requests:
            folder_info += f"node_id: {request.node_id}\nFolder Analysis:\n{request.text}\n\n"

        return f"""
        You are a software architect analyzing project folder structure. Generate comprehensive folder-level docstrings that describe the purpose, organization, and role of code directories.

        For each folder, analyze the file summaries provided and create a docstring that:
        1. **Purpose**: What specific functionality this folder encapsulates
        2. **Components**: What types of files and components are contained
        3. **Organization**: How the folder fits into the overall project structure  
        4. **Responsibilities**: Key responsibilities and domain boundaries
        5. **Usage**: How other parts of the system interact with this folder

        **Folder-Level Tags** (choose 2-4):
        - SERVICE: Microservice or major service component
        - COMPONENT: UI/Frontend component grouping
        - LAYER: Architectural layer (data, business, presentation)
        - DOMAIN: Business domain grouping  
        - INFRASTRUCTURE: Infrastructure/platform code
        - SHARED: Shared utilities/common code
        - FEATURE: Feature-specific code grouping
        - CONFIGURATION: Configuration and setup files
        - TESTING: Test suites and testing utilities
        - HELPERS: Helper functions and utilities
        - CONTROLLERS: Request/response handling
        - MODELS: Data models and structures
        - SERVICES: Business logic services
        - MIDDLEWARE: Middleware components

        IMPORTANT: You must respond with valid JSON in this exact format:
        {{
            "docstrings": [
                {{
                    "node_id": "exact_node_id_from_input",
                    "docstring": "comprehensive folder description based on contained files",
                    "tags": ["TAG1", "TAG2", "TAG3"],
                    "summary": "brief one-line summary of folder purpose"
                }}
            ]
        }}

        Folders to analyze:
        {folder_info}
        """

    def get_branch_level_prompt(self, requests: List[DocstringRequest]) -> str:
        """NEW: Generate branch-level specific prompt."""
        branch_info = ""
        for request in requests:
            branch_info += f"node_id: {request.node_id}\nBranch Context:\n{request.text}\n\n"

        return f"""
        You are a software architect analyzing project branches. Generate branch-level docstrings that describe the high-level purpose and scope of code branches.

        For each branch, provide:
        1. **Purpose**: High-level goal or theme of this branch
        2. **Scope**: What major components/areas are covered
        3. **Architecture**: Overall system architecture and patterns
        4. **Key Features**: Major functionality or capabilities
        5. **Target Users**: Who would interact with this system

        **Branch-Level Tags** (choose 2-5):
        - FEATURE: New feature development
        - RELEASE: Release branch
        - HOTFIX: Critical bug fixes
        - EXPERIMENTAL: Experimental/research code
        - INTEGRATION: Integration branch
        - MAIN: Main/production branch
        - DEVELOPMENT: Development branch
        - MAINTENANCE: Maintenance and updates
        - REFACTOR: Code refactoring
        - MIGRATION: Data/system migration

        Response format: JSON with "docstrings" array.

        Branches to analyze:
        {branch_info}
        """

    def get_function_level_prompt(self, requests: List[DocstringRequest]) -> str:
        """Enhanced function-level prompt (original logic improved)."""
        code_snippets = ""
        for request in requests:
            # Ensure we have content
            content = request.text.strip() if request.text else ""
            if not content:
                logger.warning(f"Empty content for node {request.node_id}")
                continue
                
            code_snippets += f"node_id: {request.node_id}\n```\n{content}\n```\n\n"

        if not code_snippets.strip():
            logger.error("No valid code snippets found for function-level processing")
            return ""

        return f"""
        You are a senior software engineer with expertise in code analysis and documentation. Your task is to generate concise, high-quality docstrings for each code snippet and assign meaningful tags based on its purpose. Approach this task methodically, following these steps:

        1. **Write a Concise Docstring**:
            - Begin with a clear one-line summary of the code's purpose
            - Add 1-2 additional sentences describing functionality and usage
            - Focus on being informative rather than descriptive
            - Include important technical details but avoid unnecessary implementation specifics
            - Use proper technical terminology appropriate for the codebase

        2. **Assign Specific Tags**:
            - Choose from these tag categories based on code type:

            **Backend Tags**:
            - AUTH: Authentication/authorization
            - DATABASE: Database interactions
            - API: API endpoints/handlers
            - UTILITY: Helper/utility functions
            - PRODUCER: Message producer (queues/topics)
            - CONSUMER: Message consumer (queues/topics)
            - EXTERNAL_SERVICE: External service integration
            - CONFIGURATION: Configuration management

            **Frontend Tags**:
            - UI_COMPONENT: Visual UI component
            - FORM_HANDLING: Form data handling
            - STATE_MANAGEMENT: State management
            - DATA_BINDING: Data binding
            - ROUTING: Navigation/routing
            - EVENT_HANDLING: User interaction handling
            - STYLING: Styling/theming
            - MEDIA: Media handling
            - ANIMATION: Animation
            - ACCESSIBILITY: Accessibility features
            - DATA_FETCHING: Data retrieval

            **Shared Tags**:
            - ERROR_HANDLING: Error handling/validation
            - TESTING: Testing utilities
            - SECURITY: Security mechanisms
            - PERFORMANCE: Performance optimization
            - LOGGING: Logging functionality

        Your response must be a valid JSON object containing a list of docstrings, where each docstring object has:
        - node_id: The ID of the node being documented
        - docstring: A concise description of the code's purpose and functionality
        - tags: A list of relevant tags from the categories above (choose 1-5 most relevant tags)

        Here are the code snippets:

        {code_snippets}
        """

    async def call_openai_with_prompt(self, prompt: str, node_type: str) -> List[DocstringData]:
        """NEW: Call OpenAI with type-specific prompt and retry logic (fixed error handling)."""
        
        if not prompt or prompt.strip() == "":
            logger.error(f"Empty prompt for {node_type} processing")
            return []
        
        messages = [
            {
                "role": "system", 
                "content": f"You are an expert software documentation assistant specialized in {node_type}-level analysis. Always respond with valid JSON in the exact format requested."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        retry_delay = 1
        max_retries = 3  # Reduced retries for faster debugging
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Attempting OpenAI call for {node_type} (attempt {attempt + 1})")
                
                response = await self.openai_client.chat.completions.create(
                    model=self.gpt_model,
                    messages=messages,
                    temperature=0.2,
                    response_format={"type": "json_object"},
                    max_tokens=4000 if node_type in ['file', 'folder', 'branch'] else 2000
                )
            
                content = response.choices[0].message.content.strip()
                logger.debug(f"OpenAI response for {node_type}: {content[:200]}...")
                
                parsed_response = self._parse_json_response(content)
                
                if not parsed_response.docstrings:
                    logger.error(f"No docstrings parsed from OpenAI response for {node_type}")
                    return []
                
                # Ensure node_type is set
                for docstring in parsed_response.docstrings:
                    docstring.node_type = node_type
                
                logger.debug(f"Successfully parsed {len(parsed_response.docstrings)} {node_type} docstrings")
                return parsed_response.docstrings
                
            except openai.RateLimitError:
                logger.info(f"Rate limit exceeded. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for {node_type}: {e}")
                logger.debug(f"Failed content: {content if 'content' in locals() else 'No content'}")
                return []
                
            except Exception as e:
                logger.error(f"Failed to generate {node_type} docstring: {e}")
                if attempt == max_retries - 1:  # Last attempt
                    return []
                        
        logger.error(f"Failed to generate {node_type} docstrings after {max_retries} retries")
        return []

    def _parse_json_response(self, content: str) -> DocstringResponse:
        """Enhanced JSON parser with better error handling."""
        try:
            # Clean up any potential markdown formatting
            content = re.sub(r'^```json', '', content, flags=re.MULTILINE)
            content = re.sub(r'^```', '', content, flags=re.MULTILINE)
            content = re.sub(r'```', '', content)
            content = content.strip()

            data = json.loads(content)
            docstrings = []

            for item in data.get("docstrings", []):
                if "node_id" in item and "docstring" in item:
                    # Ensure docstring is always a string
                    docstring_text = item["docstring"]
                    if not isinstance(docstring_text, str):
                        docstring_text = str(docstring_text)

                    # Ensure tags is always a list
                    tags = item.get("tags", [])
                    if isinstance(tags, str):
                        tags = [tag.strip() for tag in tags.split(",")]
                    elif not isinstance(tags, list):
                        tags = [str(tags)]

                    # Extract summary
                    summary = item.get("summary", "")
                    if not summary:
                        if "." in docstring_text:
                            summary = docstring_text.split(".", 1)[0].strip() + "."
                        else:
                            summary = docstring_text[:100]

                    docstrings.append(DocstringData(
                        node_id=item["node_id"],
                        docstring=docstring_text,
                        tags=tags,
                        node_type=item.get("node_type", "function"),
                        summary=summary,
                        complexity_score=item.get("complexity_score", 1)
                    ))

            return DocstringResponse(docstrings=docstrings)

        except Exception as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Content that failed to parse: {content[:500]}...")
            return DocstringResponse(docstrings=[])


    # ── ENHANCED: Graph Update Methods ─────────────────────────
    async def update_nodes_in_graph_hierarchical(self, repo_id: str, response: DocstringResponse) -> None:
        """ENHANCED: Update nodes with hierarchical-specific properties."""
        if not self.driver:
            raise ValueError("Neo4j connection not initialized")
            
        try:
            if not self.verify_connection():
                logger.error("Neo4j connection lost during update")
                raise ConnectionError("Neo4j connection is not active")
            
            with self.driver.session(database=self.neo4j_database) as session:
                batch_size = 25  # Smaller batches for complex updates
                total_updates = len(response.docstrings)
                total_batches = (total_updates + batch_size - 1) // batch_size
                
                logger.info(f"Updating {total_updates} hierarchical nodes in {total_batches} batches")
                
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, total_updates)
                    batch = response.docstrings[start_idx:end_idx]
                    
                    successful_updates = 0
                    failed_updates = 0
                    start_time = time.time()
                    
                    for doc in batch:
                        try:
                            # Get content based on node type
                            content = await self.get_node_content_for_embedding(session, doc)
                            
                            if not content:
                                logger.warning(f"No content found for {doc.node_type} node {doc.node_id}")
                                failed_updates += 1
                                continue
                            
                            # Generate embedding from docstring + content
                            embedding_text = f"{doc.docstring}\n{content}".strip()
                            embedding = await self.generate_embedding(embedding_text)
                            
                            if not embedding:
                                logger.error(f"Failed to generate embedding for {doc.node_type} node {doc.node_id}")
                                failed_updates += 1
                                continue
                            
                            # Ensure tags exist
                            if not doc.tags or len(doc.tags) == 0:
                                doc.tags = await self.generate_tags(embedding_text, doc.node_type)
                            
                            # Update with hierarchical properties
                            update_result = await self.update_single_node_hierarchical(session, doc, embedding)
                            
                            if update_result:
                                successful_updates += 1
                                logger.debug(f"Updated {doc.node_type} node {doc.node_id}")
                            else:
                                failed_updates += 1
                        
                        except Exception as e:
                            logger.error(f"Error updating {doc.node_type} node {doc.node_id}: {str(e)}")
                            failed_updates += 1
                    
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Hierarchical batch {batch_idx+1}/{total_batches}: "
                        f"Updated {successful_updates}/{len(batch)} nodes in {elapsed:.2f}s"
                    )
                
                logger.info(f"Hierarchical graph update complete for repository {repo_id}")
                
        except Exception as e:
            logger.error(f"Failed to update hierarchical nodes in graph: {e}")
            raise

    async def get_node_content_for_embedding(self, session, doc: DocstringData) -> str:
        """NEW: Get appropriate content for embedding based on node type."""
        try:
            if doc.node_type in ['function', 'method', 'class', 'interface']:
                result = session.run("""
                    MATCH (n) WHERE n.node_id = $node_id
                    RETURN n.content as content
                """, {"node_id": doc.node_id}).single()
                return result["content"] if result else ""
                
            elif doc.node_type == 'file':
                # For files, use summary if available, otherwise truncated content
                result = session.run("""
                    MATCH (n:FILE) WHERE n.node_id = $node_id
                    RETURN COALESCE(n.text, '') as content
                """, {"node_id": doc.node_id}).single()
                content = result["content"] if result else ""
                # Truncate large file content for embedding
                return content[:4000] + "..." if len(content) > 4000 else content
                
            elif doc.node_type == 'folder':
                # For folders, use aggregated file summaries
                return doc.summary or doc.docstring[:500]
                
            elif doc.node_type == 'branch':
                # For branches, use high-level description
                return doc.summary or doc.docstring[:500]
                
            else:
                return doc.docstring[:500]
                
        except Exception as e:
            logger.error(f"Error getting content for {doc.node_type} node {doc.node_id}: {e}")
            return doc.docstring[:500] if doc.docstring else ""

    async def update_single_node_hierarchical(self, session, doc: DocstringData, embedding: List[float]) -> bool:
        """NEW: Update a single node with hierarchical-specific properties."""
        try:
            # Prepare data - ensure all values are primitive types
            embedding_list = list(map(float, embedding))
            tags_list = list(map(str, doc.tags))
            
            # Convert complex objects to simple strings/primitives
            summary_text = str(doc.summary) if doc.summary else ""
            complexity_score = int(doc.complexity_score) if doc.complexity_score else 1
            
            # Different update queries based on node type
            if doc.node_type in ['function', 'method', 'class', 'interface']:
                query = """
                    MATCH (n) WHERE n.node_id = $node_id
                    SET n.docstring = $docstring,
                        n.embedding = $embedding,
                        n.tags = $tags,
                        n.last_updated = datetime()
                    RETURN n.docstring IS NOT NULL as success
                """
            else:
                # For file, folder, branch nodes
                query = """
                    MATCH (n) WHERE n.node_id = $node_id
                    SET n.docstring = $docstring,
                        n.embedding = $embedding,
                        n.tags = $tags,
                        n.summary = $summary,
                        n.node_type = $node_type,
                        n.last_updated = datetime()
                    RETURN n.docstring IS NOT NULL as success
                """
            
            result = session.run(query, {
                "node_id": doc.node_id,
                "docstring": str(doc.docstring),
                "embedding": embedding_list,
                "tags": tags_list,
                "summary": summary_text,
                "node_type": str(doc.node_type)
            })
            
            record = result.single()
            return record and record["success"]
            
        except Exception as e:
            logger.error(f"Error in single node update for {doc.node_id}: {e}")
            return False

    # ── PUBLIC API METHODS ─────────────────────────────────────
    async def generate_docstrings(self, repo_id: str) -> Dict[str, DocstringResponse]:
        """Main public method - delegates to hierarchical processing."""
        return await self.generate_docstrings_hierarchical(repo_id)

    def get_nodes_for_enrichment(self) -> List[Dict]:
        """Backward compatibility - get all node types."""
        all_nodes = []
        for node_type in ['function', 'method', 'class', 'interface', 'file', 'folder', 'branch']:
            nodes = self.get_nodes_for_enrichment_by_type(node_type)
            all_nodes.extend(nodes)
        return all_nodes

    async def update_nodes_in_graph(self, repo_id: str, response: DocstringResponse) -> None:
        """Backward compatibility - delegates to hierarchical update."""
        await self.update_nodes_in_graph_hierarchical(repo_id, response)

    async def process_node_batch(self, batch: List[Dict]) -> List[DocstringData]:
        """Backward compatibility - delegates to hierarchical processing."""
        return await self.process_node_batch_hierarchical(batch)

    def close(self):
        """No-op: driver is managed globally in neo4j_setup."""
        pass


# ── BACKWARD COMPATIBILITY ALIAS ───────────────────────────────
# Keep the original class name as an alias for existing code
SemanticEnrichment = HierarchicalSemanticEnrichment
