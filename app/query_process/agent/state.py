from typing_extensions import TypedDict
from typing import List, Optional
import copy


# ====================== 听书查询场景常量 ======================
SCENE_RECOMMEND = "recommend"
SCENE_DETAIL = "detail"
SCENE_RETRIEVAL = "retrieval"
SCENE_NOTES = "notes"
SCENE_UNKNOWN = "unknown"


class QueryGraphState(TypedDict, total=False):
    """
    听书知识库 - 查询流程图状态定义
    领域变更：item_names → item_names
    新增：scene（查询场景）、多字段过滤
    """
    session_id: str  # 会话唯一标识
    original_query: str  # 用户原始问题

    # 检索过程中的中间数据
    embedding_chunks: list
    hyde_embedding_chunks: list
    web_search_docs: list

    # 排序过程中的数据
    rrf_chunks: list
    reranked_docs: list

    # 生成过程中的数据
    prompt: str
    answer: str

    # ===== 听书领域字段 =====
    # 提取出的书籍名称
    item_names: List[str]
    # 提取出的作者名称
    author_names: List[str]
    # 查询场景：recommend/detail/retrieval/notes/unknown
    scene: str
    # 多字段过滤（构造好的 Milvus expr 片段）
    author_filter: str
    category_filter: str
    content_type_filter: str

    rewritten_query: str  # 改写后的问题
    history: list  # 历史对话记录
    is_stream: bool  # 是否流式输出标记


# ========================
# 默认状态（全部为空）
# ========================
query_graph_default_state: QueryGraphState = {
    "session_id": "",
    "original_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    # 听书领域
    "item_names": [],
    "author_names": [],
    "scene": SCENE_UNKNOWN,
    "author_filter": "",
    "category_filter": "",
    "content_type_filter": "",
    "rewritten_query": "",
    "history": [],
    "is_stream": False
}


# ========================
# 创建默认状态（可覆盖）
# ========================
def create_query_default_state(**overrides) -> QueryGraphState:
    """
    创建查询流程的默认状态，支持覆盖字段
    """
    state = copy.deepcopy(query_graph_default_state)
    state.update(overrides)
    return state


# ========================
# 获取干净状态
# ========================
def get_query_default_state() -> QueryGraphState:
    return copy.deepcopy(query_graph_default_state)


# ========================
# ✅ 状态复制函数（你要的）
# ========================
def copy_query_state(state: QueryGraphState, **overrides) -> QueryGraphState:
    """
    复制现有状态并可覆盖字段，深拷贝，不污染原数据
    """
    new_state = copy.deepcopy(state)
    new_state.update(overrides)
    return new_state


if __name__ == "__main__":
    # 测试
    state = create_query_default_state(
        session_id="test_001",
        original_query="三体这本书怎么样?",
        is_stream=False
    )
    print("初始化状态：", state)

    # 复制状态
    new_state = copy_query_state(
        state,
        original_query="修改后的问题"
    )
    print("复制后的状态：", new_state)