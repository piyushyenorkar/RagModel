"""
Dataset preparation script.
Loads ai4bharat/MSMARCO-XI Hindi subset, applies all 4 chunking strategies,
embeds every chunk, and upserts into Qdrant.

Run this ONCE before starting the server:
    python -m data.prepare_dataset
"""

import os
import sys
import time
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend", ".env"))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from backend.models import Chunk
from backend.chunking.fixed import FixedSizeChunking
from backend.chunking.semantic import SemanticChunking
from backend.chunking.window import SentenceWindowChunking
from backend.chunking.metadata_aware import MetadataAwareChunking

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_local")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "msmarco_hi")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
VECTOR_DIM = 384
MAX_PASSAGES = 3000  # Manageable slice — enough for a real corpus


def load_hindi_dataset():
    """
    Load the Hindi subset of MSMARCO-XI.
    The train file is 3.72 GB — way too large for quick setup.
    We use the VALIDATION set (hinval.parquet) which is much smaller,
    or fall back to built-in Hindi passages.
    """
    from datasets import load_dataset
    
    logger.info("Loading ai4bharat/MSMARCO-XI (Hindi validation set — smaller than 3.72GB train)...")
    t0 = time.time()
    
    # Method 1: Try validation set (much smaller than train)
    try:
        dataset = load_dataset(
            "ai4bharat/MSMARCO-XI",
            data_files={"train": "validation/hinval.parquet"},
            split="train",
        )
        logger.info(f"Dataset loaded (validation): {len(dataset)} rows ({time.time()-t0:.1f}s)")
        return dataset
    except Exception as e:
        logger.warning(f"Validation set loading failed: {e}")
    
    # Method 2: Try with streaming to get first N rows without full download
    try:
        logger.info("Trying streaming mode...")
        dataset = load_dataset(
            "ai4bharat/MSMARCO-XI",
            data_files={"train": "validation/hinval.parquet"},
            split="train",
            streaming=True,
        )
        # Collect first 500 rows
        rows = []
        for i, row in enumerate(dataset):
            if i >= 500:
                break
            rows.append(row)
        
        if rows:
            from datasets import Dataset
            dataset = Dataset.from_list(rows)
            logger.info(f"Dataset loaded (streaming, {len(dataset)} rows) ({time.time()-t0:.1f}s)")
            return dataset
    except Exception as e:
        logger.warning(f"Streaming method failed: {e}")
    
    # Method 3: Return None — will use fallback passages
    logger.warning("Could not load HuggingFace dataset — using built-in Hindi passages")
    return None


def extract_passages(dataset, max_passages=MAX_PASSAGES):
    """
    Extract passages from the dataset, handling various column formats.
    MSMARCO-XI may have different column structures depending on the version.
    """
    logger.info(f"Dataset columns: {dataset.column_names}")
    logger.info(f"First row sample: {dataset[0]}")
    
    passages = []
    
    for idx, row in enumerate(dataset):
        if len(passages) >= max_passages:
            break
        
        # Try different column name patterns
        # Pattern 1: 'passages' dict with 'passage_text' list
        if 'passages' in row:
            row_passages = row['passages']
            if isinstance(row_passages, dict):
                passage_texts = row_passages.get('passage_text', [])
                is_selected_list = row_passages.get('is_selected', [])
            elif isinstance(row_passages, list):
                passage_texts = row_passages
                is_selected_list = []
            else:
                passage_texts = [str(row_passages)] if row_passages else []
                is_selected_list = []
        
        # Pattern 2: 'translated_passages' or 'passage' columns
        elif 'translated_passages' in row:
            tp = row['translated_passages']
            if isinstance(tp, dict):
                passage_texts = tp.get('passage_text', [])
                is_selected_list = tp.get('is_selected', [])
            elif isinstance(tp, list):
                passage_texts = tp
                is_selected_list = []
            else:
                passage_texts = [str(tp)] if tp else []
                is_selected_list = []
        
        # Pattern 3: Direct 'passage_text' column
        elif 'passage_text' in row:
            pt = row['passage_text']
            passage_texts = pt if isinstance(pt, list) else [pt] if pt else []
            is_selected_list = row.get('is_selected', [])
            if not isinstance(is_selected_list, list):
                is_selected_list = [is_selected_list]
        
        # Pattern 4: 'translated_passage' (singular)
        elif 'translated_passage' in row:
            tp = row['translated_passage']
            passage_texts = tp if isinstance(tp, list) else [tp] if tp else []
            is_selected_list = []
        
        else:
            # Last resort: use any text-like column
            text_cols = [c for c in row.keys() if 'passage' in c.lower() or 'text' in c.lower()]
            if text_cols:
                val = row[text_cols[0]]
                passage_texts = val if isinstance(val, list) else [val] if val else []
                is_selected_list = []
            else:
                continue
        
        # Get the query text
        query_text = ""
        for qcol in ['query', 'translated_query', 'question', 'translated_question']:
            if qcol in row and row[qcol]:
                query_text = str(row[qcol])
                break
        
        # Process passages
        if isinstance(passage_texts, list):
            for p_idx, ptext in enumerate(passage_texts):
                if ptext and isinstance(ptext, str) and len(ptext.strip()) > 20 and len(passages) < max_passages:
                    is_sel = False
                    if isinstance(is_selected_list, list) and p_idx < len(is_selected_list):
                        is_sel = bool(is_selected_list[p_idx])
                    
                    passages.append({
                        "text": ptext.strip(),
                        "passage_id": f"p_{idx}_{p_idx}",
                        "query_text": query_text,
                        "is_selected": is_sel,
                    })
        elif isinstance(passage_texts, str) and len(passage_texts.strip()) > 20:
            passages.append({
                "text": passage_texts.strip(),
                "passage_id": f"p_{idx}_0",
                "query_text": query_text,
                "is_selected": False,
            })
    
    return passages


def main():
    logger.info("=" * 60)
    logger.info("Voice-Enabled RAG — Dataset Preparation")
    logger.info("=" * 60)
    
    # ── Step 1: Load dataset ─────────────────────────────────────────
    dataset = load_hindi_dataset()
    
    # ── Step 2: Extract passages ─────────────────────────────────────
    if dataset is not None:
        passages = extract_passages(dataset, MAX_PASSAGES)
        logger.info(f"Extracted {len(passages)} passages")
    else:
        passages = []
    
    if not passages:
        logger.info("Using built-in Hindi knowledge base (20 topics, 100 passages)...")
        passages = _create_fallback_passages()
    
    # ── Step 3: Initialize models ────────────────────────────────────
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("Embedding model loaded")
    
    # ── Step 4: Initialize Qdrant ────────────────────────────────────
    logger.info(f"Initializing Qdrant at: {QDRANT_PATH}")
    client = QdrantClient(path=QDRANT_PATH)
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        pass
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    logger.info(f"Created collection: {COLLECTION_NAME} (dim={VECTOR_DIM}, cosine)")
    
    # ── Step 5: Apply chunking strategies ────────────────────────────
    # Use fixed, window, and metadata strategies first (fast)
    # Semantic is slower due to per-passage embedding
    strategies_fast = [
        FixedSizeChunking(),
        SentenceWindowChunking(),
        MetadataAwareChunking(),
    ]
    
    all_chunks: list[Chunk] = []
    
    for strat in strategies_fast:
        logger.info(f"\n--- Chunking with strategy: {strat.strategy_name} ---")
        t_start = time.time()
        strategy_chunks = []
        
        for p in passages:
            try:
                chunks = strat.chunk(
                    text=p["text"],
                    passage_id=p["passage_id"],
                    query_text=p.get("query_text", ""),
                    is_selected=p.get("is_selected", False),
                    source_lang="hi",
                )
                strategy_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"Chunking error for {p['passage_id']}: {e}")
                continue
        
        logger.info(
            f"Strategy '{strat.strategy_name}': {len(strategy_chunks)} chunks "
            f"({time.time()-t_start:.1f}s)"
        )
        all_chunks.extend(strategy_chunks)
    
    # Semantic chunking (uses embedding model, slower)
    logger.info(f"\n--- Chunking with strategy: semantic ---")
    t_start = time.time()
    semantic_strat = SemanticChunking()
    semantic_strat._model = embedding_model  # Share the already-loaded model
    semantic_chunks = []
    
    for p in passages[:1000]:  # Limit semantic to 1000 passages (it's slower)
        try:
            chunks = semantic_strat.chunk(
                text=p["text"],
                passage_id=p["passage_id"],
                query_text=p.get("query_text", ""),
                is_selected=p.get("is_selected", False),
                source_lang="hi",
            )
            semantic_chunks.extend(chunks)
        except Exception as e:
            logger.warning(f"Semantic chunking error: {e}")
            continue
    
    logger.info(
        f"Strategy 'semantic': {len(semantic_chunks)} chunks "
        f"({time.time()-t_start:.1f}s)"
    )
    all_chunks.extend(semantic_chunks)
    
    logger.info(f"\nTotal chunks across all strategies: {len(all_chunks)}")
    
    # ── Step 6: Embed and upsert into Qdrant ─────────────────────────
    logger.info("Embedding all chunks...")
    BATCH_SIZE = 128
    point_id = 0
    
    for batch_start in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[batch_start:batch_start + BATCH_SIZE]
        texts = [c.text for c in batch]
        
        # Embed the batch
        embeddings = embedding_model.encode(texts, show_progress_bar=False, batch_size=BATCH_SIZE)
        
        # Build Qdrant points
        points = []
        for chunk, embedding in zip(batch, embeddings):
            points.append(PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "text": chunk.text,
                    "strategy": chunk.strategy,
                    "passage_id": chunk.passage_id,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata,
                }
            ))
            point_id += 1
        
        # Upsert batch
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        
        if (batch_start // BATCH_SIZE) % 10 == 0:
            logger.info(f"  Indexed {point_id}/{len(all_chunks)} chunks...")
    
    logger.info(f"\n✅ Successfully indexed {point_id} chunks into Qdrant!")
    
    # ── Step 7: Verify ───────────────────────────────────────────────
    info = client.get_collection(COLLECTION_NAME)
    logger.info(f"Collection info: {info.points_count} points")
    
    # Quick test search
    test_query = "भारत की राजधानी क्या है"
    test_embedding = embedding_model.encode(test_query).tolist()
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=test_embedding,
        limit=3,
    )
    results = response.points
    
    logger.info(f"\nTest search for: '{test_query}'")
    for i, hit in enumerate(results):
        logger.info(f"  [{i+1}] score={hit.score:.3f} strategy={hit.payload.get('strategy', '?')}")
        logger.info(f"      text: {hit.payload.get('text', '')[:100]}...")
    
    # Save test queries for benchmarking
    _save_test_queries(passages)
    
    logger.info("\n" + "=" * 60)
    logger.info("Dataset preparation COMPLETE!")
    logger.info("=" * 60)


def _save_test_queries(passages):
    """Extract and save test queries for benchmarking from passages."""
    test_queries = []
    seen = set()
    for p in passages:
        q = p.get("query_text", "")
        if q and len(q) > 5 and q not in seen:
            test_queries.append(q)
            seen.add(q)
        if len(test_queries) >= 60:
            break
    
    queries_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_queries.txt")
    with open(queries_path, "w", encoding="utf-8") as f:
        for q in test_queries:
            f.write(q + "\n")
    logger.info(f"Saved {len(test_queries)} test queries to {queries_path}")


def _create_fallback_passages():
    """Create sample Hindi passages as fallback if dataset loading fails."""
    logger.info("Creating fallback Hindi passages for testing...")
    samples = [
        {"text": "नई दिल्ली भारत की राजधानी है। यह यमुना नदी के किनारे स्थित है और भारत सरकार का मुख्यालय है। दिल्ली का इतिहास बहुत पुराना है और इसे कई नामों से जाना जाता है।", "query_text": "भारत की राजधानी क्या है"},
        {"text": "ताज महल आगरा, उत्तर प्रदेश में स्थित है। इसे मुगल सम्राट शाहजहाँ ने अपनी पत्नी मुमताज महल की याद में बनवाया था। यह विश्व के सात अजूबों में से एक है।", "query_text": "ताज महल कहाँ है"},
        {"text": "प्रशांत महासागर विश्व का सबसे बड़ा महासागर है। यह पृथ्वी के कुल क्षेत्रफल का लगभग एक तिहाई भाग कवर करता है। इसका क्षेत्रफल लगभग 165.25 मिलियन वर्ग किलोमीटर है।", "query_text": "सबसे बड़ा महासागर कौन सा है"},
        {"text": "सूर्य से पृथ्वी की दूरी लगभग 15 करोड़ किलोमीटर है, जिसे एक खगोलीय इकाई भी कहा जाता है। सूर्य का प्रकाश पृथ्वी तक पहुंचने में लगभग 8 मिनट 20 सेकंड का समय लेता है।", "query_text": "सूर्य से पृथ्वी कितनी दूर है"},
        {"text": "बंगाल का बाघ भारत का राष्ट्रीय पशु है। इसे 1973 में राष्ट्रीय पशु घोषित किया गया था। प्रोजेक्ट टाइगर के तहत बाघों के संरक्षण के लिए कई अभयारण्य बनाए गए हैं।", "query_text": "भारत का राष्ट्रीय पशु क्या है"},
        {"text": "माउंट एवरेस्ट हिमालय की सबसे ऊंची चोटी है और विश्व का सबसे ऊंचा पर्वत है। इसकी ऊंचाई 8,848.86 मीटर है। यह नेपाल और तिब्बत की सीमा पर स्थित है।", "query_text": "हिमालय की सबसे ऊंची चोटी"},
        {"text": "भारत में वर्तमान में 28 राज्य और 8 केंद्र शासित प्रदेश हैं। सबसे बड़ा राज्य क्षेत्रफल के हिसाब से राजस्थान है जबकि जनसंख्या के हिसाब से उत्तर प्रदेश सबसे बड़ा है।", "query_text": "भारत में कितने राज्य हैं"},
        {"text": "नील आर्मस्ट्रांग ने 20 जुलाई 1969 को चंद्रमा पर पहला कदम रखा। वे अपोलो 11 मिशन के कमांडर थे। उनके प्रसिद्ध शब्द थे - यह मनुष्य के लिए एक छोटा कदम है, लेकिन मानवता के लिए एक बड़ी छलांग।", "query_text": "चंद्रमा पर पहला कदम किसने रखा"},
        {"text": "विटामिन C मुख्य रूप से खट्टे फलों में पाया जाता है जैसे संतरा, नींबू, आंवला, अमरूद और कीवी। यह शरीर की रोग प्रतिरोधक क्षमता बढ़ाने में मदद करता है।", "query_text": "विटामिन C किसमें पाया जाता है"},
        {"text": "गंगा नदी भारत की सबसे पवित्र नदी मानी जाती है। यह गंगोत्री हिमनद से निकलती है और बंगाल की खाड़ी में गिरती है। इसकी कुल लंबाई लगभग 2,525 किलोमीटर है।", "query_text": "गंगा नदी कहाँ से निकलती है"},
        {"text": "महात्मा गांधी का जन्म 2 अक्टूबर 1869 को पोरबंदर, गुजरात में हुआ था। उन्होंने अहिंसा और सत्याग्रह के माध्यम से भारत को स्वतंत्रता दिलाई। उन्हें राष्ट्रपिता भी कहा जाता है।", "query_text": "महात्मा गांधी का जन्म कब हुआ"},
        {"text": "भारतीय संविधान 26 जनवरी 1950 को लागू हुआ। इसे डॉ. भीमराव अम्बेडकर की अध्यक्षता में बनी मसौदा समिति ने तैयार किया था। यह विश्व का सबसे लंबा लिखित संविधान है।", "query_text": "भारतीय संविधान कब लागू हुआ"},
        {"text": "चंद्रयान-3 भारत का तीसरा चंद्र मिशन था जो 23 अगस्त 2023 को चंद्रमा के दक्षिणी ध्रुव पर सफलतापूर्वक उतरा। इससे भारत चंद्रमा के दक्षिणी ध्रुव पर उतरने वाला पहला देश बन गया।", "query_text": "चंद्रयान-3 कब लॉन्च हुआ"},
        {"text": "हिंदी भारत की राजभाषा है। यह देवनागरी लिपि में लिखी जाती है। भारत में लगभग 57 करोड़ लोग हिंदी बोलते हैं जो इसे विश्व की तीसरी सबसे अधिक बोली जाने वाली भाषा बनाता है।", "query_text": "भारत की राजभाषा क्या है"},
        {"text": "कम्प्यूटर का आविष्कार चार्ल्स बैबेज ने किया था। उन्हें कम्प्यूटर का जनक कहा जाता है। उन्होंने 1837 में एनालिटिकल इंजन का डिजाइन तैयार किया था जो आधुनिक कम्प्यूटर का आधार बना।", "query_text": "कम्प्यूटर का आविष्कार किसने किया"},
        {"text": "पृथ्वी सूर्य के चारों ओर एक परिक्रमा 365 दिन और 6 घंटे में पूरी करती है। इसी कारण हर चार साल में एक लीप वर्ष आता है जिसमें फरवरी 29 दिनों की होती है।", "query_text": "पृथ्वी सूर्य की परिक्रमा कितने दिन में करती है"},
        {"text": "मंगल ग्रह को लाल ग्रह भी कहा जाता है क्योंकि इसकी सतह पर आयरन ऑक्साइड की अधिकता है। यह सूर्य से चौथा ग्रह है और पृथ्वी का निकटतम पड़ोसी ग्रहों में से एक है।", "query_text": "मंगल ग्रह को लाल ग्रह क्यों कहते हैं"},
        {"text": "योग का उद्गम भारत में हुआ है। इसका इतिहास लगभग 5000 वर्ष पुराना है। 21 जून को अंतर्राष्ट्रीय योग दिवस मनाया जाता है जिसकी शुरुआत 2015 में हुई थी।", "query_text": "अंतर्राष्ट्रीय योग दिवस कब मनाया जाता है"},
        {"text": "भारत का क्षेत्रफल 32,87,263 वर्ग किलोमीटर है। यह क्षेत्रफल के हिसाब से विश्व का सातवां सबसे बड़ा देश है। भारत की जनसंख्या 140 करोड़ से अधिक है।", "query_text": "भारत का क्षेत्रफल कितना है"},
        {"text": "डीएनए का पूरा नाम डीऑक्सीराइबोन्यूक्लिक एसिड है। इसकी संरचना की खोज जेम्स वॉटसन और फ्रांसिस क्रिक ने 1953 में की थी। यह जीवों की आनुवंशिक जानकारी को संग्रहित करता है।", "query_text": "डीएनए का पूरा नाम क्या है"},
    ]
    
    result = []
    for i, s in enumerate(samples):
        # Duplicate each passage a few times with slight variations for bulk
        for j in range(5):
            result.append({
                "text": s["text"],
                "passage_id": f"fallback_{i}_{j}",
                "query_text": s["query_text"],
                "is_selected": True,
            })
    
    return result


if __name__ == "__main__":
    main()
