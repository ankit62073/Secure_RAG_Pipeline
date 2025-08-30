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
        """ENHANCED: Main method with hierarchical processing pipeline."""
        if not self.driver:
            raise ValueError("Neo4j connection not initialized")

        try:
            self.log_graph_stats(repo_id)
            
            if not self.verify_connection():
                logger.error("Neo4j connection failed verification")
                raise ConnectionError("Neo4j connection is not active")
            
            # Process in hierarchical order: functions → files → folders → branches
            processing_order = ['function', 'method', 'class', 'interface', 'file', 'folder', 'branch']
            all_docstrings = {"docstrings": []}
            
            for node_type in processing_order:
                logger.info(f"Processing {node_type} level nodes...")
                
                nodes = self.get_nodes_for_enrichment_by_type(node_type)
                if not nodes:
                    logger.info(f"No {node_type} nodes require enrichment")
                    continue
                
                logger.info(f"Found {len(nodes)} {node_type} nodes for enrichment")
                
                # Process nodes in batches
                batch_size = self.get_optimal_batch_size(node_type)
                total_batches = (len(nodes) + batch_size - 1) // batch_size
                
                for i in range(0, len(nodes), batch_size):
                    batch = nodes[i:i + batch_size]
                    batch_num = (i // batch_size) + 1
                    
                    logger.info(f"Processing {node_type} batch {batch_num}/{total_batches} ({len(batch)} nodes)")
                    
                    try:
                        docstrings = await self.process_node_batch_hierarchical(batch)
                        
                        if docstrings:
                            await self.update_nodes_in_graph_hierarchical(repo_id, DocstringResponse(docstrings=docstrings))
                            all_docstrings["docstrings"].extend(docstrings)
                            logger.info(f"Successfully processed {len(docstrings)} {node_type} nodes")
                        
                    except Exception as e:
                        logger.error(f"Error processing {node_type} batch {batch_num}: {e}")
                        continue
            
            total_processed = len(all_docstrings["docstrings"])
            logger.info(f"Hierarchical enrichment complete: {total_processed} nodes processed across all levels")
            
            return all_docstrings

        except Exception as e:
            logger.error(f"Fatal hierarchical enrichment error: {e}")
            raise

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


import os
import json
from pathlib import Path
from tree_sitter import Parser, Node
from tree_sitter_languages import get_language
import logging
from typing import Dict, List, Any, Optional, Set, NamedTuple
from grep_ast import TreeContext
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from pygments.util import ClassNotFound

logger = logging.getLogger(__name__)

class Tag(NamedTuple):
    rel_fname: str
    fname: str
    line: int
    end_line: int
    name: str
    kind: str
    type: str

class ParsePython:
    def __init__(self):
        try:
            self.parser = Parser()
            self.parser.set_language(get_language("python"))
            self.branch_id = None
            self.warned_files = set()
                        
            logger.info("Successfully initialized parser and semantic enrichment")
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise
        
    def parse_repository(self, repo_path: str, branch_name: str) -> dict:
            """Parse all Python files in the branch and return results in-memory."""

            self.branch_id = f"{Path(repo_path).name}_{branch_name}"
            python_files = self.find_python_files(repo_path)
            parsed_data = {}

            for file_path in python_files:
                logger.info(f"Parsing file: {file_path}")
                try:
                    relative_path = str(file_path.relative_to(repo_path))
                    result = self.parse_file(file_path)
                    if result:
                        parsed_data[relative_path] = result
                except Exception as e:
                    if file_path not in self.warned_files:
                        logger.error(f"Error processing {file_path}: {e}")
                        self.warned_files.add(file_path)
                    continue

            return self.to_json_safe(parsed_data)

    def find_python_files(self, repo_path: str) -> list[Path]:
        """Find all Python files in the branch."""
        python_files = []
        for root, _, files in os.walk(repo_path):
            if any(part.startswith('.') for part in Path(root).parts):
                continue
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
                    
        logger.info(f"Found {len(python_files)} Python files in {repo_path}")
        return python_files

    def parse_file(self, file_path: Path) -> dict:
        """Parse a single Python file and extract its structure with semantic enrichment."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                source_bytes = content.encode('utf-8')
                
                # Initial attempt with tree-sitter
                try:
                    file_info = self._parse_with_tree_sitter(file_path, content, source_bytes)
                    
                    # Check if we found any functions or classes
                    if file_info and (file_info['functions'] or file_info['classes']):
                        return file_info
                    
                    logger.warning(f"No functions or classes found in {file_path} with tree-sitter, trying fallback")
                    
                except Exception as parse_error:
                    logger.warning(f"Tree-sitter parsing failed for {file_path}: {parse_error}")
                
                # First fallback: Use grep_ast for better decorator handling
                try:
                    file_info = self._parse_with_grep_ast(file_path, content)
                    if file_info and (file_info['functions'] or file_info['classes']):
                        return file_info
                        
                    logger.warning(f"No functions or classes found in {file_path} with grep_ast, trying final fallback")
                    
                except Exception as grep_error:
                    logger.warning(f"grep_ast parsing failed for {file_path}: {grep_error}")
                
                # Final fallback: Basic lexer-based parsing
                try:
                    fallback_result = self.fallback_parse(file_path, content)
                    if fallback_result:
                        fallback_result['parse_method'] = 'fallback'
                        return fallback_result
                except Exception as fallback_error:
                    logger.warning(f"Fallback parsing failed for {file_path}: {fallback_error}")
                
                # If all parsing methods fail, return minimal structure
                return {
                    'path': str(file_path),
                    'content': content,
                    'name': file_path.name,
                    'language': 'python',
                    'imports': [],
                    'classes': [],
                    'functions': [],
                    'interfaces': [],
                    'mentions': set(),
                    'branch_id': self.branch_id,
                    'parse_method': 'minimal'
                }
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None

    def _parse_with_tree_sitter(self, file_path: Path, content: str, source_bytes: bytes) -> dict:
        """Parse file using tree-sitter with enhanced decorator handling."""
        tree = self.parser.parse(source_bytes)
        
        imports = self.extract_imports(tree.root_node, source_bytes)
        classes = []
        functions = []
        interfaces = []
        
        def handle_decorated_function(node: Node) -> Optional[Dict]:
            """Process a function node that might have decorators."""
            if node.type == 'function_definition':
                func_info = self.extract_function_info(node, source_bytes)
                # Check previous siblings for decorators
                current = node
                while current.prev_sibling and current.prev_sibling.type == 'decorator':
                    decorator = source_bytes[current.prev_sibling.start_byte:current.prev_sibling.end_byte].decode('utf-8')
                    if 'decorators' not in func_info:
                        func_info['decorators'] = []
                    func_info['decorators'].append(decorator)
                    current = current.prev_sibling
                return func_info
            return None
        
        # Process each top-level node including decorators
        for node in tree.root_node.children:
            if node.type == 'class_definition':
                if self.is_interface(node, source_bytes):
                    interface_info = self.extract_interface_info(node, source_bytes)
                    interfaces.append(interface_info)
                else:
                    class_info = self.extract_class_info(node, source_bytes)
                    classes.append(class_info)
            elif node.type == 'function_definition' or (
                node.type == 'decorated_definition' and 
                node.children and 
                node.children[-1].type == 'function_definition'
            ):
                # Handle both decorated and non-decorated functions
                target_node = node.children[-1] if node.type == 'decorated_definition' else node
                function_info = handle_decorated_function(target_node)
                if function_info:
                    functions.append(function_info)
        
        file_info = {
            'path': str(file_path),
            'content': content,
            'name': file_path.name,
            'language': 'python',
            'imports': imports,
            'classes': classes,
            'functions': functions,
            'interfaces': interfaces,
            'mentions': self.extract_references(tree.root_node, source_bytes),
            'branch_id': self.branch_id,
            'parse_method': 'tree-sitter'
        }
        return file_info

    def _parse_with_grep_ast(self, file_path: Path, content: str) -> dict:
        """Parse file using grep_ast for better decorator handling."""
        # Initialize TreeContext
        context = TreeContext(str(file_path), content, color=False)
        
        # Get lexer for syntax highlighting (aids in token identification)
        lexer = guess_lexer_for_filename(str(file_path), content)
        
        classes = []
        functions = []
        imports = []
        mentions = set()
        
        # Process each line for context markers
        lines = content.splitlines()
        current_class = None
        current_function = None
        decorator_buffer = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Track decorators
            if stripped.startswith('@'):
                decorator_buffer.append(stripped)
                continue
                
            # Process class definitions
            if stripped.startswith('class '):
                class_name = stripped[6:].split('(')[0].strip()
                class_end = context.find_block_end(i)
                class_code = '\n'.join(lines[i:class_end+1])
                
                class_info = {
                    'name': class_name,
                    'code': class_code,
                    'methods': [],
                    'decorators': decorator_buffer.copy(),
                    'docstring': self._extract_docstring_from_lines(lines[i+1:class_end]),
                    'start': [i, 0],
                    'end': [class_end, len(lines[class_end])],
                }
                classes.append(class_info)
                current_class = class_info
                decorator_buffer = []
                
            # Process function definitions
            elif stripped.startswith('def '):
                func_name = stripped[4:].split('(')[0].strip()
                func_end = context.find_block_end(i)
                func_code = '\n'.join(lines[i:func_end+1])
                
                func_info = {
                    'name': func_name,
                    'code': func_code,
                    'decorators': decorator_buffer.copy(),
                    'parameters': self._extract_parameters_from_line(stripped),
                    'docstring': self._extract_docstring_from_lines(lines[i+1:func_end]),
                    'start': [i, 0],
                    'end': [func_end, len(lines[func_end])],
                }
                
                if current_class:
                    current_class['methods'].append(func_info)
                else:
                    functions.append(func_info)
                decorator_buffer = []
                
            # Track imports
            elif stripped.startswith('import ') or stripped.startswith('from '):
                imports.append({
                    'type': 'import' if stripped.startswith('import') else 'from_import',
                    'module': stripped.split()[1],
                    'start': [i, 0],
                    'end': [i, len(stripped)]
                })
            
            # Track mentions (identifiers)
            for token in lexer.get_tokens(line):
                if token[0] in Token.Name:
                    mentions.add(token[1])
        
        file_info = {
            'path': str(file_path),
            'content': content,
            'name': file_path.name,
            'language': 'python',
            'imports': imports,
            'classes': classes,
            'functions': functions,
            'interfaces': [],  # grep_ast doesn't explicitly detect interfaces
            'mentions': list(mentions),
            'branch_id': self.branch_id,
            'parse_method': 'grep_ast'
        }
        
        return file_info
        
    def _extract_parameters_from_line(self, func_def_line: str) -> List[str]:
        """Extract parameter names from a function definition line."""
        try:
            param_str = func_def_line.split('(')[1].split(')')[0]
            return [p.strip().split(':')[0].split('=')[0].strip() for p in param_str.split(',') if p.strip()]
        except:
            return []
            
    def _extract_docstring_from_lines(self, lines: List[str]) -> str:
        """Extract docstring from a sequence of lines."""
        docstring = []
        in_docstring = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if not in_docstring:
                    in_docstring = True
                    docstring.append(stripped[3:])
                else:
                    in_docstring = False
                    if len(stripped) > 3:
                        docstring.append(stripped[:-3])
                    break
            elif in_docstring:
                docstring.append(stripped)
                
        return '\n'.join(docstring).strip()

    def fallback_parse(self, file_path: Path, content: str) -> dict:
        """Fallback parsing mechanism using Pygments when tree-sitter parsing fails."""
        try:
            # Try to guess the lexer based on filename and content
            lexer = guess_lexer_for_filename(str(file_path), content)
            tokens = list(lexer.get_tokens(content))
            
            # Initialize basic structure
            file_info = {
                'path': str(file_path),
                'content': content,
                'name': file_path.name,
                'language': 'unknown',
                'imports': [],
                'classes': [],
                'functions': [],
                'interfaces': [],
                'mentions': set(),
                'branch_id': self.branch_id
            }
            
            current_block = None
            current_scope = None
            scope_stack = []
            in_class = False
            in_function = False
            current_code = []
            
            for token_type, value in tokens:
                current_code.append(value)
                
                # Track imports
                if token_type in Token.Name.Namespace:
                    if value.startswith('import ') or value.startswith('from '):
                        file_info['imports'].append({
                            'type': 'import',
                            'module': value.split()[1],
                            'alias': None
                        })
                
                # Track class definitions
                elif token_type in Token.Keyword and value == 'class':
                    in_class = True
                    if current_block:
                        scope_stack.append(current_block)
                elif in_class and token_type in Token.Name.Class:
                    current_block = {
                        'name': value,
                        'type': 'class',
                        'methods': [],
                        'code': '',
                        'docstring': '',
                        'bases': []
                    }
                    current_scope = current_block
                    in_class = False
                    file_info['classes'].append(current_block)
                
                # Track function definitions
                elif token_type in Token.Keyword and value == 'def':
                    in_function = True
                elif in_function and token_type in Token.Name.Function:
                    func_info = {
                        'name': value,
                        'code': '',
                        'docstring': '',
                        'parameters': [],
                        'calls': [],
                        'references': []
                    }
                    
                    if current_scope and 'methods' in current_scope:
                        current_scope['methods'].append(func_info)
                    else:
                        file_info['functions'].append(func_info)
                    in_function = False
                
                # Track references/mentions
                elif token_type in Token.Name:
                    file_info['mentions'].add(value)
                
                # Handle scope changes
                elif token_type in Token.Punctuation:
                    if value == '{' or value == ':':
                        if current_block:
                            scope_stack.append(current_block)
                    elif value == '}':
                        if scope_stack:
                            current_block = scope_stack.pop()
                            if scope_stack:
                                current_scope = scope_stack[-1]
                            else:
                                current_scope = None
            
            # Convert mentions set to list for JSON serialization
            file_info['mentions'] = list(file_info['mentions'])
            
            # Clean up and finalize code blocks
            for class_info in file_info['classes']:
                class_info['code'] = ''.join(current_code)
            for func_info in file_info['functions']:
                func_info['code'] = ''.join(current_code)
            
            # Enrich the fallback parsed data
            return file_info
            
        except ClassNotFound:
            logger.warning(f"Could not determine lexer for {file_path}")
            return None
        except Exception as e:
            logger.error(f"Fallback parsing failed for {file_path}: {e}")
            return None

    def is_interface(self, node: Node, source_bytes: bytes) -> bool:
        """Check if a class is an interface (abstract base class)."""
        class_code = source_bytes[node.start_byte:node.end_byte].decode('utf-8')
        
        # Check for ABC inheritance or @abstractmethod usage
        return ('ABC' in class_code or 
                'metaclass=ABCMeta' in class_code or 
                '@abstractmethod' in class_code or 
                '@abc.abstractmethod' in class_code)

    def extract_interface_info(self, node: Node, source_bytes: bytes) -> Dict:
        """Extract interface (abstract base class) information."""
        name_node = next(child for child in node.children if child.type == 'identifier')
        interface_name = source_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')
        
        # Get interface code and docstring
        interface_code = source_bytes[node.start_byte:node.end_byte].decode('utf-8')
        docstring = self._extract_docstring(node, source_bytes)
        
        # Extract abstract methods
        abstract_methods = []
        for child in node.children:
            if child.type == 'block':
                for method in child.children:
                    if (method.type == 'function_definition' and 
                        self._is_abstract_method(method, source_bytes)):
                        abstract_methods.append(self.extract_function_info(method, source_bytes))
        
        return {
            'name': interface_name,
            'qualified_name': self.get_qualified_name(name_node, source_bytes),
            'code': interface_code,
            'type': 'INTERFACE',
            'docstring': docstring,
            'abstract_methods': abstract_methods,
            'start': list(node.start_point),  # Convert tuple to list for JSON
            'end': list(node.end_point),  # Convert tuple to list for JSON
            'branch_id': self.branch_id
        }

    def _is_abstract_method(self, node: Node, source_bytes: bytes) -> bool:
        """Check if a method is abstract."""
        method_code = source_bytes[node.start_byte:node.end_byte].decode('utf-8')
        return ('@abstractmethod' in method_code or 
                '@abc.abstractmethod' in method_code or
                'raise NotImplementedError' in method_code)

    def _extract_docstring(self, node: Node, source_bytes: bytes) -> str:
        """Extract docstring from a node, generate using GPT-4o-mini if not present."""
        # First try to find an existing docstring
        for child in node.children:
            if child.type == 'block':
                for statement in child.children:
                    if statement.type == 'expression_statement':
                        expr = statement.children[0]
                        if expr.type == 'string':
                            return source_bytes[expr.start_byte:expr.end_byte].decode('utf-8').strip('"\' ')

        return ""

    def extract_imports(self, node: Node, source_bytes: bytes) -> List[Dict]:
        """Extract detailed import information."""
        imports = []
        
        def process_import_node(node: Node) -> Dict:
            if node.type == 'import_statement':
                for child in node.children:
                    if child.type == 'dotted_name':
                        name = source_bytes[child.start_byte:child.end_byte].decode('utf-8')
                        imports.append({
                            'type': 'import',
                            'module': name,
                            'alias': None,
                            'start': child.start_point,
                            'end': child.end_point
                        })
            elif node.type == 'import_from_statement':
                module = None
                for child in node.children:
                    if child.type == 'dotted_name' and module is None:
                        module = source_bytes[child.start_byte:child.end_byte].decode('utf-8')
                    elif child.type == 'dotted_name':
                        name = source_bytes[child.start_byte:child.end_byte].decode('utf-8')
                        imports.append({
                            'type': 'from_import',
                            'module': module,
                            'name': name,
                            'start': child.start_point,
                            'end': child.end_point
                        })
        
        def visit_node(node: Node):
            if node.type in ['import_statement', 'import_from_statement']:
                process_import_node(node)
            for child in node.children:
                visit_node(child)
                
        visit_node(node)
        return imports

    def extract_function_info(self, node: Node, source_bytes: bytes) -> Dict:
        """Extract detailed function information including calls and dependencies."""
        name_node = next(child for child in node.children if child.type == 'identifier')
        func_name = source_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')
        
        # Get function code
        func_code = source_bytes[node.start_byte:node.end_byte].decode('utf-8')
        
        # Extract parameters
        params = []
        param_list = next(child for child in node.children if child.type == 'parameters')
        for param in param_list.children:
            if param.type == 'identifier':
                params.append(source_bytes[param.start_byte:param.end_byte].decode('utf-8'))
        
        # Find function calls and references
        calls = []
        references = set()
        for child in node.children:
            if child.type == 'block':
                calls.extend(self.extract_calls(child, source_bytes))
                references.update(self.extract_references(child, source_bytes))
                
        return {
            'name': func_name,
            'qualified_name': self.get_qualified_name(name_node, source_bytes),
            'code': func_code,
            'docstring': self._extract_docstring(node, source_bytes),
            'parameters': params,
            'calls': calls,
            'references': list(references),
            'start': list(node.start_point),
            'end': list(node.end_point)
        }

    def extract_class_info(self, node: Node, source_bytes: bytes) -> Dict:
        """Extract detailed class information."""
        name_node = next(child for child in node.children if child.type == 'identifier')
        class_name = source_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')
        
        # Get class code
        class_code = source_bytes[node.start_byte:node.end_byte].decode('utf-8')
        
        # Get inheritance information
        bases = []
        for child in node.children:
            if child.type == 'argument_list':
                for base in child.children:
                    if base.type == 'identifier':
                        bases.append(source_bytes[base.start_byte:base.end_byte].decode('utf-8'))
        
        # Extract methods
        methods = []
        for child in node.children:
            if child.type == 'block':
                for method in child.children:
                    if method.type == 'function_definition':
                        method_info = self.extract_function_info(method, source_bytes)
                        methods.append(method_info)
        
        return {
            'name': class_name,
            'qualified_name': self.get_qualified_name(name_node, source_bytes),
            'code': class_code,
            'bases': bases,
            'methods': methods,
            'docstring': self._extract_docstring(node, source_bytes),
            'start': list(node.start_point),
            'end': list(node.end_point),
            'branch_id': self.branch_id
        }

    def extract_calls(self, node: Node, source_bytes: bytes) -> List[Dict]:
        """Extract function calls from a node."""
        calls = []
        
        def visit_calls(node: Node):
            if node.type == 'call':
                function_node = node.children[0]
                if function_node.type == 'identifier':
                    calls.append({
                        'name': source_bytes[function_node.start_byte:function_node.end_byte].decode('utf-8'),
                        'start': list(node.start_point),
                        'end': list(node.end_point)
                    })
            for child in node.children:
                visit_calls(child)
                
        visit_calls(node)
        return calls

    def extract_references(self, node: Node, source_bytes: bytes) -> Set[str]:
        """Extract all identifier references from a node."""
        references = set()
        
        def visit_references(node: Node):
            if node.type == 'identifier':
                references.add(source_bytes[node.start_byte:node.end_byte].decode('utf-8'))
            for child in node.children:
                visit_references(child)
                
        visit_references(node)
        return references

    def get_qualified_name(self, node: Node, source_bytes: bytes) -> str:
        """Get the fully qualified name of a node."""
        components = []
        current = node
        while current.parent:
            if current.type == 'identifier':
                components.insert(0, source_bytes[current.start_byte:current.end_byte].decode('utf-8'))
            current = current.parent
        return '.'.join(components)
    
    def get_tags(self, file_path: Path, relative_path: str) -> List[Tag]:
        """Extract tags from file using ctags."""
        try:
            return []  # For now, return empty list as tags are optional
        except Exception as e:
            logger.error(f"Failed to extract tags from {file_path}: {e}")
            return []

    def to_json_safe(self, data: Any) -> Any:
        """Convert any data structure to JSON-safe format."""
        if isinstance(data, (set, tuple)):
            return list(data)
        elif isinstance(data, dict):
            return {k: self.to_json_safe(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.to_json_safe(item) for item in data]
        elif isinstance(data, Path):
            return str(data)
        return data

from pathlib import Path
from tree_sitter import Parser, Node
from tree_sitter_languages import get_language
from typing import Dict, List, Any, Set
from pygments.lexers import guess_lexer_for_filename
from grep_ast import TreeContext
from pygments.token import Token
from pygments.util import ClassNotFound
import logging
import json

logger = logging.getLogger(__name__)

class ExtendedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, tuple):
            return list(obj)
        return super().default(obj)

class ParseCpp:
    def __init__(self):
        try:
            self.parser = Parser()
            self.parser.set_language(get_language("cpp"))
            self.branch_id = None
            self.warned_files = set()
            logger.info("Successfully initialized C++ parser")
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise

    def find_cpp_files(self, repo_path: str) -> List[Path]:
        """Find all C++ files in the branch."""
        cpp_files = []
        repo_path = Path(repo_path)
        
        for ext in ['.cpp', '.hpp', '.h']:
            cpp_files.extend(
                path for path in repo_path.rglob(f'*{ext}')
                if not any(part.startswith('.') for part in path.parts)
                and path.is_file()
            )
        
        logger.info(f"Found {len(cpp_files)} C++ files")
        return cpp_files

    def parse_repository(self, repo_path: str, branch_name: str) -> dict:
            """Parse all C++ files in the branch and return results in-memory."""
            
            self.branch_id = f"{Path(repo_path).name}_{branch_name}"

            cpp_files = self.find_cpp_files(repo_path)
            parsed_data = {}

            for file_path in cpp_files:
                logger.info(f"Parsing file: {file_path}")
                try:
                    file_data = self.parse_file(file_path)
                    if file_data:
                        relative_path = str(file_path.relative_to(repo_path))
                        parsed_data[relative_path] = self.to_json_safe(file_data)
                except Exception as e:
                    if file_path not in self.warned_files:
                        logger.warning(f"Failed to parse {file_path}: {e}")
                        self.warned_files.add(file_path)
                    continue

            return parsed_data

    def parse_file(self, file_path: Path) -> dict:
        """Parse a single C++ file and extract its structure."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                source_bytes = content.encode('utf-8')
                
                # Initial attempt with tree-sitter
                try:
                    parsed_data = self._parse_with_tree_sitter(file_path, content, source_bytes)
                    if parsed_data:
                        return parsed_data
                except Exception as parse_error:
                    logger.warning(f"Tree-sitter parsing failed for {file_path}: {parse_error}")
                
                # First fallback: Use grep_ast
                try:
                    parsed_data = self._parse_with_grep_ast(file_path, content)
                    if parsed_data:
                        return parsed_data
                except Exception as grep_error:
                    logger.warning(f"grep_ast parsing failed for {file_path}: {grep_error}")
                
                # Final fallback: Basic lexer-based parsing
                try:
                    parsed_data = self.fallback_parse(file_path, content)
                    if parsed_data:
                        return parsed_data
                except Exception as fallback_error:
                    logger.warning(f"Fallback parsing failed for {file_path}: {fallback_error}")
                
                # If all parsing methods fail, return minimal structure
                return {
                    'path': str(file_path),
                    'content': content,
                    'name': file_path.name,
                    'language': 'cpp',
                    'includes': [],
                    'classes': [],
                    'functions': [],
                    'structs': [],
                    'namespaces': [],
                    'mentions': set(),
                    'branch_id': self.branch_id,
                    'parse_method': 'minimal'
                }
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None

    def _parse_with_tree_sitter(self, file_path: Path, content: str, source_bytes: bytes) -> dict:

        tree = self.parser.parse(source_bytes)
        
        includes = self.extract_includes(tree.root_node, source_bytes)
        logger.info(f"Extracted includes from {file_path}: {includes}")
        classes = []
        functions = []
        structs = []
        namespaces = []
                        
        def visit_node(node: Node):
            if node.type == 'class_specifier':
                class_info = self.extract_class_info(node, source_bytes)
                if class_info:
                    classes.append(class_info)
                    
            if node.type == 'enum_specifier':
                class_info = self.extract_enum_class_info(node, source_bytes)
                if class_info:
                    classes.append(class_info)
            
            if node.type == 'function_definition':
                func_info = self.extract_function_info(node, source_bytes)
                if func_info:
                    functions.append(func_info)
            
            if node.type == 'struct_specifier':
                struct_info = self.extract_struct_info(node, source_bytes)
                if struct_info:
                    structs.append(struct_info)
            
            if node.type == 'namespace_definition':
                namespace_info = self.extract_namespace_info(node, source_bytes)
                if namespace_info:
                    namespaces.append(namespace_info)

            for child in node.children:
                visit_node(child)

        visit_node(tree.root_node)

        return {
            'path': str(file_path),
            'content': content,
            'name': file_path.name,
            'language': 'cpp',
            'includes': includes,
            'classes': classes,
            'functions': functions,
            'structs': structs,
            'namespaces': namespaces,
            'mentions': self.extract_references(tree.root_node, source_bytes),
            'branch_id': self.branch_id,
            'parse_method': 'tree-sitter'
        }

    def extract_includes(self, node: Node, source_bytes: bytes) -> List[str]:
        includes = []

        def get_text(n: Node) -> str:
            return source_bytes[n.start_byte:n.end_byte].decode("utf-8")

        def visit_node(n: Node):
            if n.type == 'preproc_include':
                # Only extract from string_literal (user-defined headers)
                header_node = next((c for c in n.children if c.type == 'string_literal'), None)

                if header_node:
                    content_node = next((c for c in header_node.children if c.type == 'string_content'), None)
                    if content_node:
                        includes.append(get_text(content_node))

            for child in n.children:
                visit_node(child)

        visit_node(node)
        return includes

    def extract_function_info(self, node: Node, source_bytes: bytes) -> Dict:
        """Extract detailed function information."""
        name = ""
        params = []

        # Iterate through the function definition node
        for child in node.children:
            if child.type == 'function_declarator':
                for subchild in child.children:
                    if subchild.type == 'identifier':
                        name = source_bytes[subchild.start_byte:subchild.end_byte].decode('utf-8')
                    if subchild.type == 'parameter_list':
                        params = self.extract_parameters(subchild, source_bytes)

        if not name:
            return None

        return {
            'name': name,
            'type': 'function',
            'parameters': params,
            'body': source_bytes[node.start_byte:node.end_byte].decode('utf-8'),
            'start': list(node.start_point),
            'end': list(node.end_point)
        }


    def extract_class_info(self, node: Node, source_bytes: bytes) -> Dict:

        name = ""
        methods = []

        for child in node.children:
            if child.type == 'type_identifier':
                name = source_bytes[child.start_byte:child.end_byte].decode('utf-8')

            elif child.type == 'field_declaration_list':
                for field in child.children:
                    if field.type == 'function_definition':
                        method_info = self.extract_function_info(field, source_bytes)
                        if method_info:
                            methods.append(method_info)

        if not name:
            return None

        return {
            'name': name,
            'type': 'class',
            'methods': methods,
            'body': source_bytes[node.start_byte:node.end_byte].decode('utf-8'),
            'start': list(node.start_point),
            'end': list(node.end_point)
        }


    def extract_struct_info(self, node: Node, source_bytes: bytes) -> Dict:
        name = ""

        for child in node.children:
            if child.type == 'type_identifier':
                name = source_bytes[child.start_byte:child.end_byte].decode('utf-8')
                break  # Only need the first identifier

        if not name:
            return None

        return {
            'name': name,
            'type': 'struct',
            'body': source_bytes[node.start_byte:node.end_byte].decode('utf-8'),
            'start': list(node.start_point),
            'end': list(node.end_point)
        }


    def extract_namespace_info(self, node: Node, source_bytes: bytes) -> Dict:
        """Extract namespace information with contained declarations."""
        name = ""
        declarations = []

        for child in node.children:
            if child.type == 'namespace_identifier':
                name = source_bytes[child.start_byte:child.end_byte].decode('utf-8')

            elif child.type == 'declaration_list':
                for decl in child.children:
                    if decl.type == 'function_definition':
                        info = self.extract_function_info(decl, source_bytes)
                        if info:
                            declarations.append(info)

                    elif decl.type == 'class_specifier':
                        info = self.extract_class_info(decl, source_bytes)
                        if info:
                            declarations.append(info)

                    elif decl.type == 'struct_specifier':
                        info = self.extract_struct_info(decl, source_bytes)
                        if info:
                            declarations.append(info)

        return {
            'name': name,
            'type': 'namespace',
            'declarations': declarations,
            'body': source_bytes[node.start_byte:node.end_byte].decode('utf-8'),  
            'start': list(node.start_point),
            'end': list(node.end_point)
        }
        
    def extract_enum_class_info(self, node: Node, source_bytes: bytes) -> Dict:
        name = ""
        enumerators = []

        for child in node.children:
            if child.type == 'type_identifier':
                name = source_bytes[child.start_byte:child.end_byte].decode('utf-8')
            elif child.type == 'enumerator_list':
                for enum_child in child.children:
                    if enum_child.type == 'enumerator':
                        enum_name = source_bytes[enum_child.start_byte:enum_child.end_byte].decode('utf-8')
                        enumerators.append(enum_name)

        if not name:
            return None

        return {
            'name': name,
            'type': 'enum_class',
            'enumerators': enumerators,
            'body': source_bytes[node.start_byte:node.end_byte].decode('utf-8'),
            'start': list(node.start_point),
            'end': list(node.end_point)
        }

    def extract_parameters(self, node: Node, source_bytes: bytes) -> List[Dict]:
        """Extract parameters from parameter_list node."""
        parameters = []

        for child in node.children:
            if child.type == 'parameter_declaration':
                param_name = ""

                for subchild in child.children:
                    if subchild.type == 'identifier':
                        param_name = source_bytes[subchild.start_byte:subchild.end_byte].decode('utf-8')

                parameters.append(param_name)

        return parameters


    def extract_references(self, node: Node, source_bytes: bytes) -> Set[str]:
        """Extract variable and type references."""
        references = set()
        
        def visit_node(node: Node):
            if node.type in ['identifier', 'type_identifier']:
                references.add(source_bytes[node.start_byte:node.end_byte].decode('utf-8'))
            for child in node.children:
                visit_node(child)
        
        visit_node(node)
        return references

    def _parse_with_grep_ast(self, file_path: Path, content: str) -> dict:
        """Parse file using grep_ast as a fallback."""
        try:
            context = TreeContext()
            tree = context.parse(str(file_path))
            
            classes = []
            functions = []
            structs = []
            namespaces = []
            includes = []
            
            for node in tree.get_definitions():
                if node.kind == 'class':
                    class_info = self._extract_class_from_grep_ast(node)
                    if class_info:
                        classes.append(class_info)
                elif node.kind == 'struct':
                    struct_info = self._extract_struct_from_grep_ast(node)
                    if struct_info:
                        structs.append(struct_info)
                elif node.kind == 'function':
                    func_info = self._extract_function_from_grep_ast(node)
                    if func_info:
                        functions.append(func_info)
                elif node.kind == 'namespace':
                    namespace_info = self._extract_namespace_from_grep_ast(node)
                    if namespace_info:
                        namespaces.append(namespace_info)
            
            return {
                'path': str(file_path),
                'content': content,
                'name': file_path.name,
                'language': 'cpp',
                'includes': includes,
                'classes': classes,
                'functions': functions,
                'structs': structs,
                'namespaces': namespaces,
                'mentions': set(),
                'branch_id': self.branch_id,
                'parse_method': 'grep_ast'
            }
            
        except Exception as e:
            logger.warning(f"grep_ast parsing failed for {file_path}: {e}")
            return None

    def _extract_class_from_grep_ast(self, node) -> Dict:
        """Extract class information from grep_ast node."""
        return {
            'name': node.name,
            'type': 'class',
            'methods': self._extract_methods_from_grep_ast(node),
            'members': self._extract_members_from_grep_ast(node),
            'start': [node.start.line, node.start.col],
            'end': [node.end.line, node.end.col]
        }

    def _extract_struct_from_grep_ast(self, node) -> Dict:
        """Extract struct information from grep_ast node."""
        return {
            'name': node.name,
            'type': 'struct',
            'members': self._extract_members_from_grep_ast(node),
            'start': [node.start.line, node.start.col],
            'end': [node.end.line, node.end.col]
        }

    def _extract_function_from_grep_ast(self, node) -> Dict:
        """Extract function information from grep_ast node."""
        return {
            'name': node.name,
            'type': 'function',
            'return_type': node.type if hasattr(node, 'type') else None,
            'parameters': self._extract_params_from_grep_ast(node),
            'start': [node.start.line, node.start.col],
            'end': [node.end.line, node.end.col]
        }

    def _extract_namespace_from_grep_ast(self, node) -> Dict:
        """Extract namespace information from grep_ast node."""
        declarations = []
        for child in node.children:
            if child.kind == 'class':
                declarations.append(self._extract_class_from_grep_ast(child))
            elif child.kind == 'struct':
                declarations.append(self._extract_struct_from_grep_ast(child))
            elif child.kind == 'function':
                declarations.append(self._extract_function_from_grep_ast(child))
        
        return {
            'name': node.name,
            'type': 'namespace',
            'declarations': declarations,
            'start': [node.start.line, node.start.col],
            'end': [node.end.line, node.end.col]
        }

    def _extract_methods_from_grep_ast(self, node) -> List[Dict]:
        """Extract methods from a grep_ast class node."""
        methods = []
        for child in node.children:
            if child.kind == 'method':
                methods.append({
                    'name': child.name,
                    'type': 'method',
                    'parameters': self._extract_params_from_grep_ast(child),
                    'start': [child.start.line, child.start.col],
                    'end': [child.end.line, child.end.col]
                })
        return methods

    def _extract_members_from_grep_ast(self, node) -> List[Dict]:
        """Extract members from a grep_ast class/struct node."""
        members = []
        for child in node.children:
            if child.kind == 'field':
                members.append({
                    'name': child.name,
                    'type': child.type if hasattr(child, 'type') else None,
                    'start': [child.start.line, child.start.col],
                    'end': [child.end.line, child.end.col]
                })
        return members

    def _extract_params_from_grep_ast(self, node) -> List[str]:
        """Extract parameters from a grep_ast function or method node."""
        params = []
        for child in node.children:
            if child.kind == 'parameter':
                params.append(child.name)
        return params

    def fallback_parse(self, file_path: Path, content: str) -> dict:
        """Basic lexer-based parsing as a last resort."""
        try:
            lexer = guess_lexer_for_filename(str(file_path), content)
            tokens = list(lexer.get_tokens(content))
            
            functions = []
            classes = []
            structs = []
            namespaces = []
            includes = []
            
            current_scope = None
            current_access = 'private'
            
            for i, (token_type, value) in enumerate(tokens):
                if token_type in Token.Keyword and value in ['class', 'struct', 'namespace']:
                    # Get the name from the next identifier token
                    for j in range(i + 1, min(i + 5, len(tokens))):
                        if tokens[j][0] in Token.Name:
                            name = tokens[j][1]
                            if value == 'class':
                                classes.append({
                                    'name': name,
                                    'type': 'class',
                                    'methods': [],
                                    'members': []
                                })
                            elif value == 'struct':
                                structs.append({
                                    'name': name,
                                    'type': 'struct',
                                    'members': []
                                })
                            elif value == 'namespace':
                                namespaces.append({
                                    'name': name,
                                    'type': 'namespace',
                                    'declarations': []
                                })
                            break
                elif token_type in Token.Keyword and value == '#include':
                    # Get the next string token for include path
                    for j in range(i + 1, min(i + 3, len(tokens))):
                        if tokens[j][0] in Token.String or tokens[j][0] in Token.Literal:
                            includes.append({
                                'path': tokens[j][1],
                                'system': tokens[j][1].startswith('<')
                            })
                            break
            
            return {
                'path': str(file_path),
                'content': content,
                'name': file_path.name,
                'language': 'cpp',
                'includes': includes,
                'classes': classes,
                'functions': functions,
                'structs': structs,
                'namespaces': namespaces,
                'mentions': set(),
                'branch_id': self.branch_id,
                'parse_method': 'lexer'
            }
            
        except ClassNotFound as e:
            logger.warning(f"Lexer not found for {file_path}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Fallback parsing failed for {file_path}: {e}")
            return None

    def to_json_safe(self, obj: Any) -> Any:
        """Convert objects to JSON-safe types."""
        if isinstance(obj, (set, Path)):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: self.to_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.to_json_safe(x) for x in obj]
        return obj
    

import os
import json
import logging
from pathlib import Path
from tree_sitter import Parser, Node
from tree_sitter_languages import get_language
from typing import Dict, List, Any, Optional, Set
from grep_ast import TreeContext
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token

logger = logging.getLogger(__name__)

class ParseJava:
    def __init__(self):
        """Initialize Java parser and Neo4j connection."""
        try:
            self.parser = Parser()
            self.parser.set_language(get_language("java"))
            self.branch_id = None
            self.warned_files = set()
            logger.info("Successfully initialized Java parser and semantic enrichment")
        except Exception as e:
            logger.error(f"Failed to initialize Java parser: {e}")
            raise

    def parse_repository(self, repo_path: str, branch_name: str) -> dict:
            """Parse all Java files in the branch and return results in-memory."""

            self.branch_id = f"{Path(repo_path).name}_{branch_name}"       

            java_files = self.find_java_files(repo_path)
            parsed_data = {}

            for file_path in java_files:
                logger.info(f"Parsing Java file: {file_path}")
                try:
                    relative_path = str(file_path.relative_to(repo_path))
                    result = self.parse_file(file_path)
                    if result:
                        parsed_data[relative_path] = result
                except Exception as e:
                    logger.error(f"Failed to parse {file_path}: {e}")

            return self.to_json_safe(parsed_data)

    def find_java_files(self, repo_path: str) -> list[Path]:
        """Find all Java files in the branch."""
        java_files = []
        for root, _, files in os.walk(repo_path):
            if any(part.startswith('.') for part in Path(root).parts):
                continue
            for file in files:
                if file.endswith('.java'):
                    java_files.append(Path(root) / file)

        logger.info(f"Found {len(java_files)} Java files in the branch.")
        return java_files

    def parse_file(self, file_path: Path) -> Optional[Dict]:
        """Parse a single Java file and extract its structure."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                source_bytes = content.encode('utf-8')
                
                # Parse with tree-sitter
                file_info = self._parse_with_tree_sitter(file_path, content, source_bytes)
                if file_info:
                    return file_info
                
                logger.warning(f"Tree-sitter parsing failed for {file_path}, trying fallback")
                return self.fallback_parse(file_path, content)
                
        except Exception as e:
            logger.error(f"Error processing Java file {file_path}: {e}")
            return None

    def _parse_with_tree_sitter(self, file_path: Path, content: str, source_bytes: bytes) -> Optional[Dict]:
        """Parse file using tree-sitter with Java-specific features."""
        tree = self.parser.parse(source_bytes)
        
        # Extract basic file info
        imports = self.extract_imports(tree.root_node, source_bytes)
        package = self.extract_package(tree.root_node, source_bytes)
        classes = []
        interfaces = []
        enums = []
        annotations = []
        
        # Process each top-level node
        for node in tree.root_node.children:
            if node.type == 'class_declaration':
                classes.append(self.extract_class_info(node, source_bytes))
            elif node.type == 'interface_declaration':
                interfaces.append(self.extract_interface_info(node, source_bytes))
            elif node.type == 'enum_declaration':
                enums.append(self.extract_enum_info(node, source_bytes))
            elif node.type == 'annotation_type_declaration':
                annotations.append(self.extract_annotation_info(node, source_bytes))
        
        file_info = {
            'path': str(file_path),
            'content': content,
            'name': file_path.name,
            'language': 'java',
            'package': package,
            'imports': imports,
            'classes': classes,
            'interfaces': interfaces,
            'enums': enums,
            'annotations': annotations,
            'mentions': self.extract_references(tree.root_node, source_bytes),
            'branch_id': self.branch_id,
            'parse_method': 'tree-sitter'
        }
        return file_info

    def extract_package(self, node: Node, source_bytes: bytes) -> Optional[Dict]:
        """Extract package declaration."""
        for child in node.children:
            if child.type == 'package_declaration':
                name_node = next((n for n in child.children if n.type == 'scoped_identifier'), None)
                if name_node:
                    return {
                        'name': source_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8'),
                        'body': source_bytes[child.start_byte:child.end_byte].decode('utf-8'),
                        'start': list(child.start_point),
                        'end': list(child.end_point)
                    }
        return None

    def extract_imports(self, node: Node, source_bytes: bytes) -> List[Dict]:
        """Extract import statements."""
        imports = []
        for child in node.children:
            if child.type == 'import_declaration':
                name_node = next((n for n in child.children if n.type == 'scoped_identifier'), None)
                if name_node:
                    static = any(n.type == 'static' for n in child.children)
                    imports.append({
                        'path': source_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8'),
                        'is_static': static,
                        'start': list(child.start_point),
                        'end': list(child.end_point)
                    })
        return imports

    def extract_class_info(self, node: Node, source_bytes: bytes) -> Dict:
        """Extract class information including methods, fields, and modifiers."""
        name_node = next(n for n in node.children if n.type == 'identifier')
        class_name = source_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')
        
        # Get class modifiers
        modifiers = []
        for child in node.children:
            if child.type == 'modifiers':
                modifiers = [source_bytes[mod.start_byte:mod.end_byte].decode('utf-8') 
                           for mod in child.children]
                
        # Get superclass and interfaces
        extends = None
        implements = []
        for child in node.children:
            if child.type == 'superclass':
                type_node = next((n for n in child.children if n.type == 'type_identifier'), None)
                if type_node:
                    extends = source_bytes[type_node.start_byte:type_node.end_byte].decode('utf-8')
            elif child.type == 'interfaces':
                for interface in child.children:
                    if interface.type == 'type_identifier':
                        implements.append(source_bytes[interface.start_byte:interface.end_byte].decode('utf-8'))
                        
        methods = []
        fields = []
        class_body_str = ""
        for child in node.children:
            if child.type == 'class_body':
                class_body_str = source_bytes[child.start_byte:child.end_byte].decode('utf-8')
                for member in child.children:
                    if member.type == 'method_declaration':
                        methods.append(self.extract_method_info(member, source_bytes))
                    elif member.type == 'field_declaration':
                        fields.append(self.extract_field_info(member, source_bytes))
        
        # Get methods and fields
        methods = []
        fields = []
        for child in node.children:
            if child.type == 'class_body':
                for member in child.children:
                    if member.type == 'method_declaration':
                        methods.append(self.extract_method_info(member, source_bytes))
                    elif member.type == 'field_declaration':
                        fields.append(self.extract_field_info(member, source_bytes))
        
        return {
            'name': class_name,
            'qualified_name': self.get_qualified_name(name_node, source_bytes),
            'modifiers': modifiers,
            'extends': extends,
            'implements': implements,
            'methods': methods,
            'fields': fields,
            'docstring': self._extract_javadoc(node, source_bytes),
            'body': class_body_str,
            'start': list(node.start_point),
            'end': list(node.end_point)
        }

    def extract_interface_info(self, node: Node, source_bytes: bytes) -> Dict:
        """Extract interface information."""
        name_node = next(n for n in node.children if n.type == 'identifier')
        interface_name = source_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')
        
        # Get modifiers
        modifiers = []
        for child in node.children:
            if child.type == 'modifiers':
                modifiers = [source_bytes[mod.start_byte:mod.end_byte].decode('utf-8') 
                           for mod in child.children]
        
        # Get extended interfaces
        extends = []
        for child in node.children:
            if child.type == 'interfaces':
                for interface in child.children:
                    if interface.type == 'type_identifier':
                        extends.append(source_bytes[interface.start_byte:interface.end_byte].decode('utf-8'))
        
        # Get methods
        methods = []
        interface_body_str = ""
        for child in node.children:
            if child.type == 'interface_body':
                interface_body_str = source_bytes[child.start_byte:child.end_byte].decode('utf-8')
                for member in child.children:
                    if member.type == 'method_declaration':
                        methods.append(self.extract_method_info(member, source_bytes))
        
        return {
            'name': interface_name,
            'qualified_name': self.get_qualified_name(name_node, source_bytes),
            'modifiers': modifiers,
            'extends': extends,
            'methods': methods,
            'docstring': self._extract_javadoc(node, source_bytes),
            'body': interface_body_str,
            'start': list(node.start_point),
            'end': list(node.end_point)
        }

    def extract_method_info(self, node: Node, source_bytes: bytes) -> Dict:
        """Extract method information."""
        name_node = next(n for n in node.children if n.type == 'identifier')
        method_name = source_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')
        
        # Get modifiers
        modifiers = []
        for child in node.children:
            if child.type == 'modifiers':
                modifiers = [source_bytes[mod.start_byte:mod.end_byte].decode('utf-8') 
                           for mod in child.children]
        
        # Get return type
        return_type = None
        type_node = next((n for n in node.children if n.type in ['type_identifier', 'void_type']), None)
        if type_node:
            return_type = source_bytes[type_node.start_byte:type_node.end_byte].decode('utf-8')
        
        # Get parameters
        parameters = []
        param_list = next((n for n in node.children if n.type == 'formal_parameters'), None)
        if param_list:
            for param in param_list.children:
                if param.type == 'formal_parameter':
                    param_type = next((n for n in param.children if n.type == 'type_identifier'), None)
                    param_name = next((n for n in param.children if n.type == 'identifier'), None)
                    if param_type and param_name:
                        parameters.append({
                            'type': source_bytes[param_type.start_byte:param_type.end_byte].decode('utf-8'),
                            'name': source_bytes[param_name.start_byte:param_name.end_byte].decode('utf-8')
                        })
        
        # Get method body and extract calls
        body = None
        calls = []
        for child in node.children:
            if child.type == 'block':
                body = source_bytes[child.start_byte:child.end_byte].decode('utf-8')
                calls = self.extract_calls(child, source_bytes)
        
        return {
            'name': method_name,
            'qualified_name': self.get_qualified_name(name_node, source_bytes),
            'modifiers': modifiers,
            'return_type': return_type,
            'parameters': parameters,
            'body': body,
            'calls': calls,
            'docstring': self._extract_javadoc(node, source_bytes),
            'start': list(node.start_point),
            'end': list(node.end_point)
        }

    def extract_field_info(self, node: Node, source_bytes: bytes) -> Dict:
        """Extract field information."""
        # Get field type
        type_node = next((n for n in node.children if n.type == 'type_identifier'), None)
        field_type = source_bytes[type_node.start_byte:type_node.end_byte].decode('utf-8') if type_node else None
        
        # Get field name and initializer
        declarator = next((n for n in node.children if n.type == 'variable_declarator'), None)
        if declarator:
            name_node = next((n for n in declarator.children if n.type == 'identifier'), None)
            initializer = next((n for n in declarator.children if n.type == 'initializer'), None)
            
            return {
                'name': source_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8') if name_node else None,
                'type': field_type,
                'initializer': source_bytes[initializer.start_byte:initializer.end_byte].decode('utf-8') if initializer else None,
                'modifiers': [source_bytes[mod.start_byte:mod.end_byte].decode('utf-8') 
                            for mod in node.children if mod.type == 'modifiers'],
                'docstring': self._extract_javadoc(node, source_bytes),
                'start': list(node.start_point),
                'end': list(node.end_point)
            }
        return None

    def extract_enum_info(self, node: Node, source_bytes: bytes) -> Dict:
        """Extract enum information."""
        name_node = next(n for n in node.children if n.type == 'identifier')
        enum_name = source_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')
        
        # Get enum constants
        constants = []
        body = next((n for n in node.children if n.type == 'enum_body'), None)
        if body:
            for const in body.children:
                if const.type == 'enum_constant':
                    const_name = next((n for n in const.children if n.type == 'identifier'), None)
                    if const_name:
                        constants.append(source_bytes[const_name.start_byte:const_name.end_byte].decode('utf-8'))
        
        return {
            'name': enum_name,
            'qualified_name': self.get_qualified_name(name_node, source_bytes),
            'constants': constants,
            'modifiers': [source_bytes[mod.start_byte:mod.end_byte].decode('utf-8') 
                         for mod in node.children if mod.type == 'modifiers'],
            'docstring': self._extract_javadoc(node, source_bytes),
            'start': list(node.start_point),
            'end': list(node.end_point)
        }

    def extract_annotation_info(self, node: Node, source_bytes: bytes) -> Dict:
        """Extract annotation type information."""
        name_node = next(n for n in node.children if n.type == 'identifier')
        annotation_name = source_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')
        
        # Get annotation elements
        elements = []
        body = next((n for n in node.children if n.type == 'annotation_type_body'), None)
        if body:
            for elem in body.children:
                if elem.type == 'annotation_type_element_declaration':
                    elem_name = next((n for n in elem.children if n.type == 'identifier'), None)
                    if elem_name:
                        elements.append({
                            'name': source_bytes[elem_name.start_byte:elem_name.end_byte].decode('utf-8'),
                            'type': self._get_annotation_element_type(elem, source_bytes)
                        })
        
        return {
            'name': annotation_name,
            'qualified_name': self.get_qualified_name(name_node, source_bytes),
            'elements': elements,
            'modifiers': [source_bytes[mod.start_byte:mod.end_byte].decode('utf-8') 
                         for mod in node.children if mod.type == 'modifiers'],
            'docstring': self._extract_javadoc(node, source_bytes),
            'start': list(node.start_point),
            'end': list(node.end_point)
        }

    def _get_annotation_element_type(self, node: Node, source_bytes: bytes) -> str:
        """Get the type of an annotation element."""
        type_node = next((n for n in node.children if n.type == 'type_identifier'), None)
        if type_node:
            return source_bytes[type_node.start_byte:type_node.end_byte].decode('utf-8')
        return 'unknown'

    def extract_calls(self, node: Node, source_bytes: bytes) -> List[Dict]:
        """Extract method calls from a node."""
        calls = []
        
        def visit_calls(node: Node):
            if node.type == 'method_invocation':
                name_node = next((n for n in node.children if n.type == 'identifier'), None)
                if name_node:
                    calls.append({
                        'name': source_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8'),
                        'line': name_node.start_point[0],
                        'code': source_bytes[node.start_byte:node.end_byte].decode('utf-8')
                    })
            for child in node.children:
                visit_calls(child)
                
        visit_calls(node)
        return calls

    def extract_references(self, node: Node, source_bytes: bytes) -> Set[str]:
        """Extract all identifier references from a node."""
        references = set()
        
        def visit_references(node: Node):
            if node.type == 'identifier':
                references.add(source_bytes[node.start_byte:node.end_byte].decode('utf-8'))
            for child in node.children:
                visit_references(child)
                
        visit_references(node)
        return references

    def get_qualified_name(self, node: Node, source_bytes: bytes) -> str:
        """Get the fully qualified name of a node."""
        components = []
        current = node
        
        while current.parent:
            if current.type == 'identifier':
                components.insert(0, source_bytes[current.start_byte:current.end_byte].decode('utf-8'))
            elif current.type == 'package_declaration':
                package_node = next((n for n in current.children if n.type == 'scoped_identifier'), None)
                if package_node:
                    components.insert(0, source_bytes[package_node.start_byte:package_node.end_byte].decode('utf-8'))
            current = current.parent
            
        return '.'.join(components)

    def _extract_javadoc(self, node: Node, source_bytes: bytes) -> Optional[str]:
        try:
            # Get parent node since comments are siblings
            parent = node.parent
            if not parent:
                return None
                
            # Get all named siblings that come before this node
            siblings = []
            for child in parent.children:
                if child == node:
                    break
                if child.is_named:
                    siblings.append(child)
                    
            # Check siblings in reverse order to get closest comment
            for sibling in reversed(siblings):
                if sibling.type in ('line_comment', 'block_comment'):
                    comment = source_bytes[sibling.start_byte:sibling.end_byte].decode('utf-8')
                    if comment.startswith('/**'):
                        return comment.strip()
                        
            return None
            
        except Exception as e:
            logger.error(f"Error extracting Javadoc: {e}")
            return None    

    def fallback_parse(self, file_path: Path, content: str) -> Optional[Dict]:
        """Fallback parsing using grep-ast when tree-sitter fails."""
        try:
            # Initialize TreeContext for block detection
            context = TreeContext(str(file_path), content, color=False)
            
            # Get lexer for token identification
            lexer = guess_lexer_for_filename(str(file_path), content)
            
            # Initialize file info structure
            file_info = {
                'path': str(file_path),
                'content': content,
                'name': file_path.name,
                'language': 'java',
                'package': None,
                'imports': [],
                'classes': [],
                'interfaces': [],
                'enums': [],
                'annotations': [],
                'mentions': set(),
                'branch_id': self.branch_id,
                'parse_method': 'grep-ast'
            }
            
            # Process each line for context markers
            lines = content.splitlines()
            current_class = None
            current_interface = None
            current_enum = None
            current_annotation = None
            decorator_buffer = []
            in_block = False
            access_specifier = "private"  # Default access in Java
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                
                # Skip empty lines and comments
                if not stripped or stripped.startswith('//') or stripped.startswith('/*'):
                    continue
                
                # Handle annotations/decorators
                if stripped.startswith('@'):
                    decorator_buffer.append(stripped)
                    continue
                
                # Handle package declaration
                if stripped.startswith('package '):
                    package_name = stripped[8:].strip(';')
                    file_info['package'] = {
                        'name': package_name,
                        'start': [i, 0],
                        'end': [i, len(line)]
                    }
                    continue
                
                # Handle imports
                if stripped.startswith('import '):
                    is_static = 'static ' in stripped
                    import_path = stripped[7:].replace('static ', '').strip(';')
                    file_info['imports'].append({
                        'path': import_path,
                        'is_static': is_static,
                        'start': [i, 0],
                        'end': [i, len(line)]
                    })
                    continue
                  # Handle access modifiers
                if stripped.startswith(('public ', 'private ', 'protected ')):
                    access_specifier = stripped.split()[0]
                    continue
                
                # Handle class declarations
                if self._is_class_declaration(stripped):
                    class_end = context.find_block_end(i)
                    if class_end:
                        class_info = self._parse_class_declaration(stripped, i)
                        if class_info:
                            class_info['body'] = '\n'.join(lines[i:class_end+1])
                            class_info['decorators'] = decorator_buffer.copy()
                            class_info['methods'] = self._grep_methods('\n'.join(lines[i:class_end+1]))
                            class_info['fields'] = self._grep_fields('\n'.join(lines[i:class_end+1]))
                            class_info['end'] = [class_end, len(lines[class_end])]
                            file_info['classes'].append(class_info)
                            current_class = class_info
                            decorator_buffer = []
                
                # Handle interface declarations
                elif stripped.startswith('interface '):
                    interface_name = stripped[10:].split('{')[0].split('extends')[0].strip()
                    interface_end = context.find_block_end(i)
                    if interface_end:
                        interface_info = {
                            'name': interface_name,
                            'type': 'interface',
                            'body': '\n'.join(lines[i:interface_end+1]),
                            'methods': [],
                            'decorators': decorator_buffer.copy(),
                            'extends': [],
                            'start': [i, 0],
                            'end': [interface_end, len(lines[interface_end])]
                        }
                        
                        # Extract extends clause
                        if 'extends ' in stripped:
                            extends_list = stripped.split('extends ')[1].split('{')[0].strip().split(',')
                            interface_info['extends'] = [ext.strip() for ext in extends_list]
                        
                        # Extract method signatures
                        interface_info['methods'] = self._grep_methods('\n'.join(lines[i:interface_end+1]))
                        file_info['interfaces'].append(interface_info)
                        current_interface = interface_info
                        decorator_buffer = []
                
                # Handle enum declarations
                elif stripped.startswith('enum '):
                    enum_name = stripped[5:].split('{')[0].strip()
                    enum_end = context.find_block_end(i)
                    if enum_end:
                        enum_code = '\n'.join(lines[i:enum_end+1])
                        enum_info = {
                            'name': enum_name,
                            'type': 'enum',
                            'body': enum_code,
                            'values': [],
                            'methods': [],
                            'decorators': decorator_buffer.copy(),
                            'start': [i, 0],
                            'end': [enum_end, len(lines[enum_end])]
                        }
                        
                        # Extract enum values
                        enum_body = enum_code[enum_code.find('{')+1:enum_code.rfind('}')].strip()
                        value_section = enum_body.split(';')[0] if ';' in enum_body else enum_body
                        values = [v.strip() for v in value_section.split(',') if v.strip()]
                        enum_info['values'] = values
                        
                        # Extract any methods defined in the enum
                        if ';' in enum_body:
                            enum_info['methods'] = self._grep_methods(enum_body[enum_body.find(';')+1:])
                        
                        file_info['enums'].append(enum_info)
                        current_enum = enum_info
                        decorator_buffer = []
                
                # Handle annotation declarations
                elif stripped.startswith('@interface '):
                    ann_name = stripped[11:].split('{')[0].strip()
                    ann_end = context.find_block_end(i)
                    if ann_end:
                        ann_code = '\n'.join(lines[i:ann_end+1])
                        ann_info = {
                            'name': ann_name,
                            'type': 'annotation',
                            'body': ann_code,
                            'elements': [],
                            'decorators': decorator_buffer.copy(),
                            'start': [i, 0],
                            'end': [ann_end, len(lines[ann_end])]
                        }
                        
                        # Extract annotation elements (methods)
                        ann_info['elements'] = self._grep_methods(ann_code)
                        file_info['annotations'].append(ann_info)
                        current_annotation = ann_info
                        decorator_buffer = []
                  # Track identifier mentions
                for token in lexer.get_tokens(line):
                    if token[0] in Token.Name:
                        if not any(c.isupper() for c in token[1][1:]):  # Skip camelCase/PascalCase
                            file_info['mentions'].add(token[1])
            
            # Convert mentions set to list for JSON serialization
            file_info['mentions'] = list(file_info['mentions'])
            
            return file_info
            
        except Exception as e:
            logger.error(f"Fallback parsing failed for {file_path}: {e}")
            return None

    def _grep_package(self, content: str) -> Optional[Dict]:
        """Extract package declaration using grep-ast."""
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('package '):
                package_name = line.strip()[8:].strip(';')
                return {
                    'name': package_name,
                    'start': [i, line.find('package')],
                    'end': [i, len(line.rstrip())]
                }
        return None

    def _grep_imports(self, content: str) -> List[Dict]:
        """Extract import statements using grep-ast."""
        imports = []
        lines = content.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('import '):
                is_static = 'static ' in line
                import_path = line.replace('import ', '').replace('static ', '').strip(';')
                imports.append({
                    'path': import_path,
                    'is_static': is_static,
                    'start': [i, 0],
                    'end': [i, len(line)]
                })
        return imports

    def _grep_classes(self, content: str) -> List[Dict]:
        """Extract class declarations and their members using grep-ast."""
        classes = []
        lines = content.split('\n')
        current_class = None
        in_class = False
        class_content = []
        brace_count = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Handle class declaration
            if not in_class and self._is_class_declaration(stripped):
                in_class = True
                brace_count = stripped.count('{') - stripped.count('}')
                class_content = [line]
                class_info = self._parse_class_declaration(stripped, i)
                if class_info:
                    current_class = class_info
                continue
                
            if in_class:
                class_content.append(line)
                brace_count += stripped.count('{') - stripped.count('}')
                
                if brace_count == 0:
                    in_class = False
                    if current_class:
                        current_class['body'] = '\n'.join(class_content)
                        current_class['methods'] = self._grep_methods('\n'.join(class_content))
                        current_class['fields'] = self._grep_fields('\n'.join(class_content))
                        classes.append(current_class)
                        current_class = None
                        class_content = []
                        
        return classes

    def _is_class_declaration(self, line: str) -> bool:
        """Check if a line contains a class declaration."""
        patterns = ['class ', 'public class ', 'private class ', 'protected class ']
        return any(pattern in line and '{' in line for pattern in patterns)

    def _parse_class_declaration(self, line: str, line_num: int) -> Optional[Dict]:
        """Parse a class declaration line."""
        try:
            # Extract class name and modifiers
            parts = line.split('class ')[0].split()
            modifiers = parts[:-1] if parts else []
            class_def = line.split('class ')[1]
            class_name = class_def.split('{')[0].split('extends')[0].split('implements')[0].strip()
            
            # Extract extends and implements
            extends = None
            implements = []
            if 'extends ' in class_def:
                extends = class_def.split('extends ')[1].split('implements')[0].split('{')[0].strip()
            if 'implements ' in class_def:
                impls = class_def.split('implements ')[1].split('{')[0].strip()
                implements = [i.strip() for i in impls.split(',')]
            
            return {
                'name': class_name,
                'modifiers': modifiers,
                'extends': extends,
                'implements': implements,
                'methods': [],
                'fields': [],
                'start': [line_num, 0],
                'end': [line_num, len(line)]
            }
        except Exception:
            return None

    def _grep_methods(self, content: str) -> List[Dict]:
        """Extract method declarations from class content."""
        methods = []
        lines = content.split('\n')
        in_method = False
        method_content = []
        brace_count = 0
        current_method = None
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip if we're still in a previous method
            if not in_method and self._is_method_declaration(stripped):
                in_method = True
                brace_count = stripped.count('{') - stripped.count('}')
                method_content = [line]
                method_info = self._parse_method_declaration(stripped, i)
                if method_info:
                    current_method = method_info
                continue
                
            if in_method:
                method_content.append(line)
                brace_count += stripped.count('{') - stripped.count('}')
                
                if brace_count == 0:
                    in_method = False
                    if current_method:
                        current_method['body'] = '\n'.join(method_content)
                        methods.append(current_method)
                        current_method = None
                        method_content = []
                    
            return methods

    def _is_method_declaration(self, line: str) -> bool:
        """Check if a line contains a method declaration."""
        if not line or '{' not in line:
            return False
        if '(' not in line or ')' not in line:
            return False
            
        method_patterns = [
            'public ', 'private ', 'protected ', 'void ', 'static ',
            'final ', 'synchronized ', 'abstract ', 'native '
        ]
        # Check if it starts with a modifier or has a known Java return type
        parts = line.split('(')[0].split()
        if not parts:
            return False
            
        has_modifier = any(part in ['public', 'private', 'protected', 'static', 'final', 'synchronized', 'abstract', 'native'] 
                          for part in parts)
        has_type = any(part in ['void', 'boolean', 'byte', 'char', 'short', 'int', 'long', 'float', 'double', 'String'] 
                      or part.endswith('[]') for part in parts)
        
        return has_modifier or has_type or line.split('(')[0].count('.') == 0  # Constructor check    
    
    def _parse_method_declaration(self, line: str, line_num: int) -> Optional[Dict]:
        """Parse a method declaration line."""
        try:
            # Extract method name and declaration parts
            before_paren = line.split('(')[0].strip()
            parts = before_paren.split()
            method_name = parts[-1]
            modifiers = []
            return_type = None
            type_params = None
            
            # Extract method name with any generic type parameters
            if '<' in method_name and '>' in method_name:
                method_name = method_name[:method_name.find('<')]
                
            # Process parts before method name
            for part in parts[:-1]:
                if part in ['public', 'private', 'protected', 'static', 'final', 'abstract', 'synchronized', 'native']:
                    modifiers.append(part)
                else:
                    return_type = part
                    if '<' in part and '>' in part:
                        type_params = self._parse_generic_type_params(part)
            
            # Extract parameters
            params_str = line[line.index('('):line.index(')')].strip('()')        
            parameters = []
            if params_str:
                for param in params_str.split(','):
                    param = param.strip()
                    if param:
                        param_parts = param.split()
                        param_type = ' '.join(param_parts[:-1])
                        param_name = param_parts[-1]
                        
                        # Handle array types
                        if '[]' in param_name:
                            param_type = param_type + '[]' * param_name.count('[]')
                            param_name = param_name.replace('[]', '')
                            
                        # Handle generic types in parameters
                        param_type_params = self._parse_generic_type_params(param_type)
                        
                        parameters.append({
                            'type': param_type.split('<')[0] if param_type_params else param_type,
                            'name': param_name,
                            'type_params': param_type_params,
                            'is_final': 'final' in param_parts[:-1]
                        })
                
            return {
                'name': method_name,
                'modifiers': modifiers,
                'return_type': return_type,
                'type_params': type_params,
                'parameters': parameters,
                'start': [line_num, 0],
                'end': [line_num, len(line)]
            }
            
        except Exception:
            return None

    def _grep_fields(self, content: str) -> List[Dict]:
        """Extract field declarations from class content."""
        fields = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip method declarations and other non-field lines
            if (not stripped or
                    '{' in stripped or
                    stripped.startswith('//') or
                    stripped.startswith('/*') or
                    stripped.startswith('*') or
                    '(' in stripped):
                continue
            
            # Parse potential field declaration
            if ';' in stripped:
                field_info = self._parse_field_declaration(stripped, i)
                if field_info:
                    fields.append(field_info)
                    
        return fields

    def _parse_field_declaration(self, line: str, line_num: int) -> Optional[Dict]:
        """Parse a field declaration line."""
        try:
            parts = line.strip(';').split('=')
            declaration = parts[0].strip()
            initializer = parts[1].strip() if len(parts) > 1 else None
            
            # Split declaration into modifiers, type, and name
            decl_parts = declaration.split()
            field_name = decl_parts[-1]
            modifiers = []
            field_type = None
            
            for part in decl_parts[:-1]:
                if part in ['public', 'private', 'protected', 'static', 'final', 'volatile', 'transient']:
                    modifiers.append(part)
                else:
                    field_type = part
            
            return {
                'name': field_name,
                'type': field_type,
                'modifiers': modifiers,
                'initializer': initializer,
                'start': [line_num, 0],
                'end': [line_num, len(line)]
            }
        except Exception:
            return None

    def to_json_safe(self, data: Any) -> Any:
        """Convert any data structure to JSON-safe format."""
        if isinstance(data, (set, tuple)):
            return list(data)
        elif isinstance(data, dict):
            return {k: self.to_json_safe(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.to_json_safe(item) for item in data]
        elif isinstance(data, Path):
            return str(data)
        return data

    def _parse_generic_type_params(self, type_str: str) -> Optional[str]:
        """Parse generic type parameters from a type declaration."""
        if '<' in type_str and '>' in type_str:
            generics = type_str[type_str.find('<')+1:type_str.rfind('>')]
            return generics.strip()
        return None

import os
import json
from pathlib import Path
from tree_sitter import Parser, Node
from tree_sitter_languages import get_language
import logging
from typing import Dict, List, Set
from grep_ast import TreeContext
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from pygments.util import ClassNotFound

logger = logging.getLogger(__name__)

class ExtendedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, tuple):
            return list(obj)
        return super().default(obj)

class ParseJavaScript:
    def __init__(self):
        try:
            self.parser = Parser()
            self.parser.set_language(get_language("javascript"))
            self.branch_id = None
            self.warned_files = set()
            logger.info("Successfully initialized JavaScript parser")
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise

    def find_javascript_files(self, repo_path: str) -> List[Path]:
        """Find all JavaScript files in the branch."""
        js_files = []
        repo_path = Path(repo_path)
        
        for ext in ['.js', '.jsx']:
            js_files.extend(
                path for path in repo_path.rglob(f'*{ext}')
                if not any(part.startswith('.') for part in path.parts)
                and path.is_file()
            )
        
        logger.info(f"Found {len(js_files)} JavaScript files")
        return js_files

    def parse_repository(self, repo_path: str, branch_name: str) -> dict:
            """Parse all JavaScript files in the branch and return results in-memory."""

            self.branch_id = f"{Path(repo_path).name}_{branch_name}"

            js_files = self.find_javascript_files(repo_path)
            parsed_data = {}

            for file_path in js_files:
                logger.info(f"Parsing file: {file_path}")
                try:
                    file_data = self.parse_file(file_path)
                    if file_data:
                        relative_path = str(file_path.relative_to(repo_path))
                        file_data['path'] = relative_path  # Ensure path is set correctly
                        parsed_data[relative_path] = file_data
                except Exception as e:
                    if file_path not in self.warned_files:
                        logger.error(f"Error processing {file_path}: {e}")
                        self.warned_files.add(file_path)
                    continue

            return parsed_data

    def parse_file(self, file_path: Path) -> dict:
        """Parse a single javascript file and extract its structure with semantic enrichment."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                source_bytes = content.encode('utf-8')
                
                # Initial attempt with tree-sitter
                try:
                    file_info = self._parse_with_tree_sitter(file_path, content, source_bytes)
                    
                    return file_info
                    
                except Exception as parse_error:
                    logger.warning(f"Tree-sitter parsing failed for {file_path}: {parse_error}")
                
                # First fallback: Use grep_ast for better decorator handling
                try:
                    file_info = self._parse_with_grep_ast(file_path, content)
                    return file_info
                    
                except Exception as grep_error:
                    logger.warning(f"grep_ast parsing failed for {file_path}: {grep_error}")
                
                # Final fallback: Basic lexer-based parsing
                try:
                    fallback_result = self.fallback_parse(file_path, content)
                    if fallback_result:
                        fallback_result['parse_method'] = 'fallback'
                        return fallback_result
                except Exception as fallback_error:
                    logger.warning(f"Fallback parsing failed for {file_path}: {fallback_error}")
                
                # If all parsing methods fail, return minimal structure
                return {
                    'path': str(file_path),
                    'content': content,
                    'name': file_path.name,
                    'language': 'javascript',
                    'imports': [],
                    'classes': [],
                    'functions': [],
                    'interfaces': [],
                    'mentions': set(),
                    'branch_id': self.branch_id,
                    'parse_method': 'minimal'
                }
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None

    def _parse_with_tree_sitter(self, file_path: Path, content: str, source_bytes: bytes) -> dict:
        """Parse file using tree-sitter."""
        tree = self.parser.parse(source_bytes)
        
        imports = self.extract_imports(tree.root_node, source_bytes)
        exports = self.extract_exports(tree.root_node, source_bytes)
        classes = []
        functions = []
                        
        def visit_node(node: Node):
                        
            if node.type == 'lexical_declaration':
                
                for child in node.children:
                    if child.type == 'variable_declarator':
                        class_name = ""
                        class_node = None

                        for grandchild in child.children:
                            if grandchild.type == 'identifier':
                                class_name = source_bytes[grandchild.start_byte:grandchild.end_byte].decode('utf-8')
                            if grandchild.type == 'class':
                                class_node = grandchild

                        if class_name and class_node:
                            class_info = self.extract_class_info(class_node, source_bytes, override_name=class_name)
                            classes.append(class_info)
                
                for child in node.children:
                    if child.type == 'variable_declarator':
                        function_name = ""
                        function_node = None

                        for grandchild in child.children:
                            if grandchild.type == 'identifier':
                                function_name = source_bytes[grandchild.start_byte:grandchild.end_byte].decode('utf-8')
                            if grandchild.type == 'function':
                                function_node = grandchild
                            if grandchild.type == 'arrow_function':
                                function_node = grandchild

                        if function_node and function_name:
                            func_info = self.extract_function_info(function_node, source_bytes, override_name=function_name)
                            functions.append(func_info)

            elif node.type in ['function_declaration', 'method_definition']:
                functions.append(self.extract_function_info(node, source_bytes))

            elif node.type == 'expression_statement':
                
                for child in node.children:
                    if child.type == 'assignment_expression':
                        fn_name = ""
                        fn_node = None

                        for grandchild in child.children:
                            if grandchild.type == 'identifier' or grandchild.type == 'member_expression':
                                fn_name = source_bytes[grandchild.start_byte:grandchild.end_byte].decode('utf-8')
                            if grandchild.type == 'arrow_function' or grandchild.type == 'function':
                                fn_node = grandchild

                        if fn_name and fn_node:
                            func_info = self.extract_function_info(fn_node, source_bytes, override_name=fn_name)
                            functions.append(func_info)

            elif node.type == 'arrow_function':
                functions.append(self.extract_function_info(node, source_bytes))

            for child in node.children:
                visit_node(child)

        visit_node(tree.root_node)

        return {
            'path': str(file_path),
            'content': content,
            'name': file_path.name,
            'language': 'javascript',
            'imports': imports,
            'exports': exports,
            'classes': classes,
            'functions': functions,
            'interfaces': [],
            'mentions': self.extract_references(tree.root_node, source_bytes),
            'branch_id': self.branch_id,
            'parse_method': 'tree-sitter'
        }

    def extract_imports(self, node: Node, source_bytes: bytes) -> List[Dict]:
        """Extract import and require declarations."""
        imports = []

        def get_text(n: Node) -> str:
            return source_bytes[n.start_byte:n.end_byte].decode("utf-8")

        def visit_node(n: Node):
            # ES6 Imports
            if n.type == 'lexical_declaration':
                declarator = next((c for c in n.children if c.type == 'variable_declarator'), None)
                if declarator:
                    pattern_node = next((c for c in declarator.children if c.type in ['object_pattern', 'identifier']), None)
                    call_expr = next((c for c in declarator.children if c.type == 'call_expression'), None)

                    if call_expr:
                        callee = next((c for c in call_expr.children if c.type == 'identifier'), None)
                        arg = next((c for c in call_expr.children if c.type == 'arguments'), None)

                        if callee and get_text(callee) == 'require' and arg:
                            string_node = next((c for c in arg.children if c.type == 'string'), None)
                            if string_node:
                                module_path = get_text(string_node).strip('"\'')
                                specifiers = []

                                if pattern_node and pattern_node.type == 'object_pattern':
                                    for p in pattern_node.children:
                                        if p.type == 'shorthand_property_identifier_pattern':
                                            specifiers.append(get_text(p))

                                elif pattern_node and pattern_node.type == 'identifier':
                                    specifiers.append(get_text(pattern_node))

                                imports.append({
                                    'type': 'require',
                                    'source': module_path,
                                    'specifiers': specifiers,
                                    'start': list(n.start_point),
                                    'end': list(n.end_point),
                                })

            for child in n.children:
                visit_node(child)

        visit_node(node)
        return imports

    def extract_exports(self, node: Node, source_bytes: bytes) -> List[Dict]:
        """Extract CommonJS export declarations like `module.exports = { ... }`."""
        exports = []

        def visit_node(node: Node):
            if node.type == 'expression_statement':
                # Check for an assignment of the form: module.exports = {...}
                expr = next((c for c in node.children if c.type == 'assignment_expression'), None)
                if expr and len(expr.children) >= 3:
                    left = expr.children[0]
                    if left.type == 'member_expression':
                        left_text = source_bytes[left.start_byte:left.end_byte].decode('utf-8').strip()
                        if left_text == 'module.exports':
                            exported_names = []
                            right = expr.children[2]
                            if right.type == 'object':
                                for prop in right.named_children:
                                    if prop.type == 'shorthand_property_identifier':
                                        name = source_bytes[prop.start_byte:prop.end_byte].decode('utf-8')
                                        exported_names.append(name)
                            exports.append({
                                'type': 'module.exports',
                                'exports': exported_names,
                                'start': list(node.start_point),
                                'end': list(node.end_point)
                            })

            for child in node.children:
                visit_node(child)

        visit_node(node)
        return exports

    def extract_function_info(self, node: Node, source_bytes: bytes, override_name: str = "") -> Dict:
        """Extract detailed function information, including arrow or normal type."""
        name = override_name
        
        params = []
        function_type = "normal_function"  # default

        if node.type == 'arrow_function':
            function_type = "arrow_function"

        for child in node.children:
            if child.type == 'identifier':
                name = source_bytes[child.start_byte:child.end_byte].decode('utf-8')
            if child.type == 'formal_parameters':
                for param in child.children:
                    if param.type == 'identifier':
                        params.append(source_bytes[param.start_byte:param.end_byte].decode('utf-8'))
        
        if not name:
            return None

        return {
            'name': name,
            'type': function_type,
            'parameters': params,
            'body': source_bytes[node.start_byte:node.end_byte].decode('utf-8'),
            'start': list(node.start_point),
            'end': list(node.end_point)
        }

    def extract_class_info(self, node: Node, source_bytes: bytes, override_name: str = "") -> Dict:
        """Extract detailed class information."""
        name = override_name
        methods = []
        
        for child in node.children:
            if child.type == 'identifier' and not name:
                name = source_bytes[child.start_byte:child.end_byte].decode('utf-8')
            if child.type == 'class_body':
                for class_element in child.children:
                    if class_element.type == 'method_definition':
                        method_info = self.extract_method_info(class_element, source_bytes)
                        if method_info:
                            methods.append(method_info)
                    if class_element.type == 'public_field_definition':
                        # Extract field information if needed
                        pass
        
        if not name:
            return None
        
        return {
            'name': name,
            'type': 'class',
            'methods': methods,
            'body': source_bytes[node.start_byte:node.end_byte].decode('utf-8'),
            'start': list(node.start_point),
            'end': list(node.end_point)
        }
        
    def extract_method_info(self, node: Node, source_bytes: bytes) -> Dict:
        """Extract detailed method information."""
        name = ""
        parameters = []
        return_type = ""
        is_static = False
        access_modifier = "public"  # Default

        for child in node.children:
            if child.type == 'property_identifier':
                name = source_bytes[child.start_byte:child.end_byte].decode('utf-8')
            if child.type == 'formal_parameters':
                parameters = self.extract_parameters(child, source_bytes)
            if child.type == 'type_annotation':
                return_type = source_bytes[child.start_byte:child.end_byte].decode('utf-8')
            if child.type == 'static':
                is_static = True
            if child.type == 'accessibility_modifier':
                access_modifier = source_bytes[child.start_byte:child.end_byte].decode('utf-8')

        return {
            'name': name,
            'parameters': parameters,
            'return_type': return_type,
            'is_static': is_static,
            'access_modifier': access_modifier,
            'body': source_bytes[node.start_byte:node.end_byte].decode('utf-8'),
            'start': list(node.start_point),
            'end': list(node.end_point)
        }
    
    def extract_parameters(self, node: Node, source_bytes: bytes) -> List[Dict]:
        """Extract parameters from formal parameters node."""
        params = []
        for param in node.children:
            if param.type == 'required_parameter' or param.type == 'optional_parameter':
                param_name = ""
                param_type = ""
                for param_child in param.children:
                    if param_child.type == 'identifier':
                        param_name = source_bytes[param_child.start_byte:param_child.end_byte].decode('utf-8')
                    if param_child.type == 'type_annotation':
                        param_type = source_bytes[param_child.start_byte:param_child.end_byte].decode('utf-8')
                params.append({'name': param_name, 'type': param_type})
        return params

    def extract_references(self, node: Node, source_bytes: bytes) -> Set[str]:
        """Extract variable and function references."""
        references = set()
        
        def visit_node(node: Node):
            if node.type == 'identifier':
                references.add(source_bytes[node.start_byte:node.end_byte].decode('utf-8'))
            for child in node.children:
                visit_node(child)
        
        visit_node(node)
        return references

    def _parse_with_grep_ast(self, file_path: Path, content: str) -> dict:
        """Parse file using grep_ast as a fallback."""
        try:
            context = TreeContext()
            tree = context.parse(str(file_path))
            
            classes = []
            functions = []
            interfaces = []
            imports = []
            exports = []
            
            for node in tree.get_definitions():
                if node.kind == 'class':
                    classes.append({
                        'name': node.name,
                        'type': 'class',
                        'start': [node.start.line, node.start.col],
                        'end': [node.end.line, node.end.col],
                        'methods': self._extract_methods_from_grep_ast(node)
                    })
                elif node.kind == 'interface':
                    interfaces.append(self._extract_interface_info(node))
                elif node.kind == 'function':
                    functions.append({
                        'name': node.name,
                        'type': 'function',
                        'start': [node.start.line, node.start.col],
                        'end': [node.end.line, node.end.col],
                        'parameters': self._extract_params_from_grep_ast(node)
                    })
            
            return {
                'path': str(file_path),
                'content': content,
                'name': file_path.name,
                'language': 'javascript',
                'imports': imports,
                'exports': exports,
                'classes': classes,
                'functions': functions,
                'interfaces': interfaces,
                'mentions': set(),
                'branch_id': self.branch_id,
                'parse_method': 'grep_ast'
            }
            
        except Exception as e:
            logger.warning(f"grep_ast parsing failed for {file_path}: {e}")
            return None

    def _extract_methods_from_grep_ast(self, node) -> List[Dict]:
        """Extract methods from a grep_ast class node."""
        methods = []
        for child in node.children:
            if child.kind == 'method':
                methods.append({
                    'name': child.name,
                    'type': 'method',
                    'parameters': self._extract_params_from_grep_ast(child),
                    'start': [child.start.line, child.start.col],
                    'end': [child.end.line, child.end.col]
                })
        return methods

    def _extract_params_from_grep_ast(self, node) -> List[str]:
        """Extract parameters from a grep_ast function or method node."""
        params = []
        for child in node.children:
            if child.kind == 'parameter':
                params.append(child.name)
        return params

    def _extract_interface_info(self, node) -> Dict:
        """Extract interface information from a grep_ast interface node."""
        methods = []
        properties = []
        
        for child in node.children:
            if child.kind == 'method':
                methods.append({
                    'name': child.name,
                    'parameters': self._extract_params_from_grep_ast(child),
                    'return_type': child.type if hasattr(child, 'type') else None,
                    'start': [child.start.line, child.start.col],
                    'end': [child.end.line, child.end.col]
                })
            elif child.kind == 'property':
                properties.append({
                    'name': child.name,
                    'type': child.type if hasattr(child, 'type') else None,
                    'start': [child.start.line, child.start.col],
                    'end': [child.end.line, child.end.col]
                })
        
        return {
            'name': node.name,
            'type': 'interface',
            'methods': methods,
            'properties': properties,
            'start': [node.start.line, node.start.col],
            'end': [node.end.line, node.end.col]
        }

    def fallback_parse(self, file_path: Path, content: str) -> dict:
        """Basic lexer-based parsing as a last resort."""
        try:
            lexer = guess_lexer_for_filename(str(file_path), content)
            tokens = list(lexer.get_tokens(content))
            
            functions = []
            classes = []
            current_class = None
            current_function = None
            
            for i, (token_type, value) in enumerate(tokens):
                if token_type in Token.Name.Function:
                    # Look ahead for opening parenthesis to confirm it's a function
                    if i + 1 < len(tokens) and tokens[i + 1][1] == '(':
                        current_function = {
                            'name': value,
                            'type': 'function',
                            'parameters': [],
                            'body': '',
                            'start': [0, 0],  # Basic lexer doesn't provide line numbers
                            'end': [0, 0]
                        }
                        if current_class:
                            if 'methods' not in current_class:
                                current_class['methods'] = []
                            current_class['methods'].append(current_function)
                        else:
                            functions.append(current_function)
                
                elif token_type in Token.Keyword and value == 'class':
                    # Look ahead for class name
                    if i + 1 < len(tokens) and tokens[i + 1][0] in Token.Name:
                        current_class = {
                            'name': tokens[i + 1][1],
                            'type': 'class',
                            'methods': [],
                            'body': '',
                            'start': [0, 0],
                            'end': [0, 0]
                        }
                        classes.append(current_class)
            
            return {
                'path': str(file_path),
                'content': content,
                'name': file_path.name,
                'language': 'javascript',
                'imports': [],
                'exports': [],
                'classes': classes,
                'functions': functions,
                'interfaces': [],
                'mentions': set(),
                'branch_id': self.branch_id,
                'parse_method': 'lexer'
            }
            
        except ClassNotFound as e:
            logger.warning(f"Lexer not found for {file_path}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Fallback parsing failed for {file_path}: {e}")
            return None
        
import os
import json
import logging
from pathlib import Path
from tree_sitter import Parser, Node
from tree_sitter_languages import get_language
from typing import Dict, List, Any, Optional, Set
from grep_ast import TreeContext
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from pygments.util import ClassNotFound

logger = logging.getLogger(__name__)

class ParseKotlin:
    def __init__(self):

        try:
            self.parser = Parser()
            self.parser.set_language(get_language("kotlin"))
            self.branch_id = None
            self.warned_files = set()
            logger.info("Successfully initialized Kotlin parser and semantic enrichment")
        except Exception as e:
            logger.error(f"Failed to initialize Kotlin parser: {e}")
            raise

    def parse_repository(self, repo_path: str, branch_name: str) -> dict:
            """Parse all Kotlin files in the branch and return results in-memory."""

            self.branch_id = f"{Path(repo_path).name}_{branch_name}"

            kotlin_files = self.find_kotlin_files(repo_path)
            parsed_data = {}

            for file_path in kotlin_files:
                logger.info(f"Parsing Kotlin file: {file_path}")
                try:
                    relative_path = str(file_path.relative_to(repo_path))
                    result = self.parse_file(file_path)
                    if result:
                        parsed_data[relative_path] = result
                except Exception as e:
                    logger.error(f"Failed to parse {file_path}: {e}")

            return self.to_json_safe(parsed_data)

    def find_kotlin_files(self, repo_path: str) -> list[Path]:
        """Find all Kotlin files in the branch."""
        kotlin_files = []
        for root, _, files in os.walk(repo_path):
            if any(part.startswith('.') for part in Path(root).parts):
                continue
            for file in files:
                if file.endswith('.kt') or file.endswith('.kts'):
                    kotlin_files.append(Path(root) / file)
                    
        return kotlin_files

    def parse_file(self, file_path: Path) -> dict:
        """Parse a single Kotlin file and extract its structure with semantic enrichment."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                source_bytes = content.encode('utf-8')
                
                # Initial attempt with tree-sitter
                try:
                    file_info = self._parse_with_tree_sitter(file_path, content, source_bytes)
                    
                    return file_info
                    
                except Exception as parse_error:
                    logger.warning(f"Tree-sitter parsing failed for {file_path}: {parse_error}")
                
                # First fallback: Use grep_ast for better decorator handling
                try:
                    file_info = self._parse_with_grep_ast(file_path, content)
                    return file_info
                    
                except Exception as grep_error:
                    logger.warning(f"grep_ast parsing failed for {file_path}: {grep_error}")
                
                # Final fallback: Basic lexer-based parsing
                try:
                    fallback_result = self.fallback_parse(file_path, content)
                    if fallback_result:
                        fallback_result['parse_method'] = 'fallback'
                        return fallback_result
                except Exception as fallback_error:
                    logger.warning(f"Fallback parsing failed for {file_path}: {fallback_error}")
                
                # If all parsing methods fail, return minimal structure
                return {
                    'path': str(file_path),
                    'content': content,
                    'name': file_path.name,
                    'language': 'kotlin',
                    'imports': [],
                    'classes': [],
                    'functions': [],
                    'interfaces': [],
                    'mentions': set(),
                    'branch_id': self.branch_id,
                    'parse_method': 'minimal'
                }
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None

    def _parse_with_tree_sitter(self, file_path: Path, content: str, source_bytes: bytes) -> Optional[Dict]:
        """Parse file using tree-sitter with Kotlin-specific features."""
        tree = self.parser.parse(source_bytes)

        # Extract basic file info
        package = self.extract_package(tree.root_node, source_bytes)
        imports = self.extract_imports(tree.root_node, source_bytes)
        classes = []
        interfaces = []
        functions = []
        
        # Process each top-level node
        for node in tree.root_node.children:
            if node.type == 'class_declaration':
                classes.append(self.extract_class_info(node, source_bytes))
            if node.type == 'interface_declaration':
                interfaces.append(self.extract_interface_info(node, source_bytes))
            if node.type == 'function_declaration':
                functions.append(self.extract_function_info(node, source_bytes))
        
        file_info = {
            'path': str(file_path),
            'content': content,
            'name': file_path.name,
            'language': 'kotlin',
            'imports': imports,
            'classes': classes,
            'functions': functions,
            'packages': package,
            'interfaces': [],
            'mentions': self.extract_references(tree.root_node, source_bytes),
            'branch_id': self.branch_id,
            'parse_method': 'tree-sitter'
        }
        
        return file_info

    def extract_package(self, node: Node, source_bytes: bytes) -> Optional[Dict]:
        """Extract package declaration from Kotlin AST."""
        for child in node.children:
            if child.type == 'package_header':
                identifier_node = next((n for n in child.children if n.type == 'identifier'), None)
                if identifier_node:
                    # Collect text from all simple_identifiers
                    parts = []
                    for sub in identifier_node.children:
                        if sub.type == 'simple_identifier':
                            part = source_bytes[sub.start_byte:sub.end_byte].decode('utf-8')
                            parts.append(part)
                    if parts:
                        return {
                            'name': '.'.join(parts),
                            'start': list(child.start_point),
                            'end': list(child.end_point)
                        }
        return None

    def extract_imports(self, node: Node, source_bytes: bytes) -> List[Dict]:
        """Extract import statements from Kotlin AST."""
        imports = []
        for child in node.children:
            if child.type == 'import_list':
                for import_node in child.children:
                    if import_node.type == 'import_header':
                        identifier_node = next((n for n in import_node.children if n.type == 'identifier'), None)
                        if identifier_node:
                            parts = []
                            for sub in identifier_node.children:
                                if sub.type == 'simple_identifier':
                                    part = source_bytes[sub.start_byte:sub.end_byte].decode('utf-8')
                                    parts.append(part)
                            if parts:
                                imports.append({
                                    'path': '.'.join(parts),
                                    'is_all': any(n.type == '*' for n in import_node.children),
                                    'start': list(import_node.start_point),
                                    'end': list(import_node.end_point)
                                })
        return imports


    def extract_class_info(self, node: Node, source_bytes: bytes) -> List[Dict]:
        
        def get_text(child: Node):
            return source_bytes[child.start_byte:child.end_byte].decode('utf-8')

        class_name = None
        methods = []
        start_line = node.start_point[0]
        end_line = node.end_point[0]
        class_body = ""

        for child in node.children:
            if child.type == 'type_identifier':
                class_name = get_text(child)
            if child.type == 'class_body' or child.type == 'enum_class_body':
                for item in child.children:
                    if item.type == 'function_declaration':
                        methods.append(self.extract_function_info(item, source_bytes))
            # Optional: Capture full class content as string
            class_body = get_text(node)

        return {
            "name": class_name,
            "start_line": start_line,
            "end_line": end_line,
            "methods": methods,
            "content": class_body
        }

    def extract_interface_info(self, node: Node, source_bytes: bytes) -> Dict:
        """Extract interface information."""
        name_node = next(n for n in node.children if n.type == 'type_identifier')
        interface_name = source_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')

        # Get abstract methods and properties
        methods = []
        interface_body_str = ""

        for child in node.children:
            if child.type == 'class_body':
                interface_body_str = source_bytes[child.start_byte:child.end_byte].decode('utf-8')
                for member in child.children:
                    if member.type == 'function_declaration':
                        methods.append(self.extract_function_info(member, source_bytes))

        return {
            'name': interface_name,
            'methods': methods,
            'content': interface_body_str,
            'start': list(node.start_point),
            'end': list(node.end_point)
        }

    def extract_function_info(self, node: Node, source_bytes: bytes) -> Dict:
        """Extract function information from Kotlin AST."""

        # Extract full function text
        full_function_text = source_bytes[node.start_byte:node.end_byte].decode('utf-8')

        # Extract function name
        name_node = next((n for n in node.children if n.type == 'simple_identifier'), None)
        function_name = source_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8') if name_node else 'anonymous'

        parameters = []

        # Look for function_value_parameters and extract each parameter as raw text
        param_list = next((n for n in node.children if n.type == 'function_value_parameters'), None)
        if param_list:
            for param in param_list.children:
                if param.type == 'parameter':
                    param_text = source_bytes[param.start_byte:param.end_byte].decode('utf-8')
                    parameters.append(param_text)

        return {
            'name': function_name,
            'parameters': parameters,
            'content': full_function_text,
            'start': list(node.start_point),
            'end': list(node.end_point)
        }


    def extract_references(self, node: Node, source_bytes: bytes) -> Set[str]:
        """Extract all identifier references from a node."""
        references = set()
        
        def visit_references(node: Node):
            if node.type in ('simple_identifier', 'type_identifier'):
                references.add(source_bytes[node.start_byte:node.end_byte].decode('utf-8'))
            for child in node.children:
                visit_references(child)
                
        visit_references(node)
        return references
    
    def _parse_with_grep_ast(self, file_path: Path, content: str) -> dict:
        """Parse file using grep_ast as a fallback."""
        try:
            context = TreeContext()
            tree = context.parse(str(file_path))
            
            classes = []
            functions = []
            interfaces = []
            imports = []
            exports = []
            
            for node in tree.get_definitions():
                if node.kind == 'class':
                    classes.append({
                        'name': node.name,
                        'type': 'class',
                        'start': [node.start.line, node.start.col],
                        'end': [node.end.line, node.end.col],
                        'methods': self._extract_methods_from_grep_ast(node)
                    })
                elif node.kind == 'interface':
                    interfaces.append(self._extract_interface_info(node))
                elif node.kind == 'function':
                    functions.append({
                        'name': node.name,
                        'type': 'function',
                        'start': [node.start.line, node.start.col],
                        'end': [node.end.line, node.end.col],
                        'parameters': self._extract_params_from_grep_ast(node)
                    })
            
            return {
                'path': str(file_path),
                'content': content,
                'name': file_path.name,
                'language': 'javascript',
                'imports': imports,
                'exports': exports,
                'classes': classes,
                'functions': functions,
                'interfaces': interfaces,
                'mentions': set(),
                'branch_id': self.branch_id,
                'parse_method': 'grep_ast'
            }
            
        except Exception as e:
            logger.warning(f"grep_ast parsing failed for {file_path}: {e}")
            return None

    def _extract_methods_from_grep_ast(self, node) -> List[Dict]:
        """Extract methods from a grep_ast class node."""
        methods = []
        for child in node.children:
            if child.kind == 'method':
                methods.append({
                    'name': child.name,
                    'type': 'method',
                    'parameters': self._extract_params_from_grep_ast(child),
                    'start': [child.start.line, child.start.col],
                    'end': [child.end.line, child.end.col]
                })
        return methods

    def _extract_params_from_grep_ast(self, node) -> List[str]:
        """Extract parameters from a grep_ast function or method node."""
        params = []
        for child in node.children:
            if child.kind == 'parameter':
                params.append(child.name)
        return params

    def _extract_interface_info(self, node) -> Dict:
        """Extract interface information from a grep_ast interface node."""
        methods = []
        properties = []
        
        for child in node.children:
            if child.kind == 'method':
                methods.append({
                    'name': child.name,
                    'parameters': self._extract_params_from_grep_ast(child),
                    'return_type': child.type if hasattr(child, 'type') else None,
                    'start': [child.start.line, child.start.col],
                    'end': [child.end.line, child.end.col]
                })
            elif child.kind == 'property':
                properties.append({
                    'name': child.name,
                    'type': child.type if hasattr(child, 'type') else None,
                    'start': [child.start.line, child.start.col],
                    'end': [child.end.line, child.end.col]
                })
        
        return {
            'name': node.name,
            'type': 'interface',
            'methods': methods,
            'properties': properties,
            'start': [node.start.line, node.start.col],
            'end': [node.end.line, node.end.col]
        }

    def fallback_parse(self, file_path: Path, content: str) -> dict:
        """Basic lexer-based parsing as a last resort."""
        try:
            lexer = guess_lexer_for_filename(str(file_path), content)
            tokens = list(lexer.get_tokens(content))
            
            functions = []
            classes = []
            current_class = None
            current_function = None
            
            for i, (token_type, value) in enumerate(tokens):
                if token_type in Token.Name.Function:
                    # Look ahead for opening parenthesis to confirm it's a function
                    if i + 1 < len(tokens) and tokens[i + 1][1] == '(':
                        current_function = {
                            'name': value,
                            'type': 'function',
                            'parameters': [],
                            'body': '',
                            'start': [0, 0],  # Basic lexer doesn't provide line numbers
                            'end': [0, 0]
                        }
                        if current_class:
                            if 'methods' not in current_class:
                                current_class['methods'] = []
                            current_class['methods'].append(current_function)
                        else:
                            functions.append(current_function)
                
                elif token_type in Token.Keyword and value == 'class':
                    # Look ahead for class name
                    if i + 1 < len(tokens) and tokens[i + 1][0] in Token.Name:
                        current_class = {
                            'name': tokens[i + 1][1],
                            'type': 'class',
                            'methods': [],
                            'body': '',
                            'start': [0, 0],
                            'end': [0, 0]
                        }
                        classes.append(current_class)
            
            return {
                'path': str(file_path),
                'content': content,
                'name': file_path.name,
                'language': 'javascript',
                'imports': [],
                'exports': [],
                'classes': classes,
                'functions': functions,
                'interfaces': [],
                'mentions': set(),
                'branch_id': self.branch_id,
                'parse_method': 'lexer'
            }
            
        except ClassNotFound as e:
            logger.warning(f"Lexer not found for {file_path}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Fallback parsing failed for {file_path}: {e}")
            return None

    def to_json_safe(self, data: Any) -> Any:
        """Convert any data structure to JSON-safe format."""
        if isinstance(data, (set, tuple)):
            return list(data)
        elif isinstance(data, dict):
            return {k: self.to_json_safe(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.to_json_safe(item) for item in data]
        elif isinstance(data, Path):
            return str(data)
        return data
