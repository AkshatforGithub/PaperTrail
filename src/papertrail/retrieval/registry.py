"""One list of retrievers shared by run_eval, the CLI and the API."""

from papertrail.retrieval.hybrid import hybrid_rerank_search, hybrid_search
from papertrail.retrieval.keyword_search import keyword_search
from papertrail.retrieval.vector_search import vector_search

RETRIEVERS = {
    "vector": vector_search,
    "keyword": keyword_search,
    "hybrid": hybrid_search,
    "hybrid_rerank": hybrid_rerank_search,
}
DEFAULT_RETRIEVER = "hybrid_rerank"  # set this to whichever scores best in compare_runs
