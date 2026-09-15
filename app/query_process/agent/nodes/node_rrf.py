import sys
from typing import List, Dict, Any
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger

# ================================
# LangGraph RRF 融合节点
# 功能：接收多路向量检索结果 → 统一格式 → 加权融合 → 输出最终排序列表
# ================================

def node_rrf(state):
    logger.info("---RRF (倒数排名融合) 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # ==============================================
    # 步骤1：从 state 取出两路召回结果
    # ==============================================
    embedding_chunks = _as_entity_list(state.get("embedding_chunks"))
    hyde_embedding_chunks = _as_entity_list(state.get("hyde_embedding_chunks"))

    logger.info(f"RRF 输入统计: Embedding源={len(embedding_chunks)}条, HyDE源={len(hyde_embedding_chunks)}条")

    # Debug：打印前5条ID便于核对
    if embedding_chunks:
        logger.debug(f"Embedding源 chunk_ids (前5个): {[c.get('chunk_id') for c in embedding_chunks[:5]]}")
    if hyde_embedding_chunks:
        logger.debug(f"HyDE源 chunk_ids (前5个): {[c.get('chunk_id') for c in hyde_embedding_chunks[:5]]}")

    # ==============================================
    # 步骤2：配置多路权重（可根据业务调整）
    # ==============================================
    source_weights = [
        (embedding_chunks, 1.0),
        (hyde_embedding_chunks, 1.0)
    ]

    # ==============================================
    # 步骤3：执行 RRF 融合排序
    # ==============================================
    rrf_res = reciprocal_rank_fusion(source_weights, k=60, max_results=10)

    # ==============================================
    # 步骤4：提取最终文档列表
    # ==============================================
    rrf_chunks = [doc for doc, score in rrf_res]

    # 任务完成标记
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 把融合结果存入 state
    return {"rrf_chunks": rrf_chunks}

# ================================
# 工具函数：统一格式化检索结果
# 功能：将不同来源（Milvus Hit/字典/自定义对象）统一转为标准实体列表
# ================================
def _as_entity_list(state_list) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for doc in (state_list or []):
        if not doc:
            continue

        final_ent = {}

        # ==============================================
        # 情况A：处理 Milvus 返回的 Hit 对象（含 entity、id、distance）
        # ==============================================
        if hasattr(doc, "entity") and hasattr(doc, "id"):
            # 提取 entity 内容（支持对象转字典 / 直接是字典）
            entity_content = doc.entity
            if hasattr(entity_content, "to_dict"):
                final_ent = entity_content.to_dict()
            elif isinstance(entity_content, dict):
                final_ent = entity_content.copy()
            else:
                # 尝试强转字典，兼容不同 SDK 版本
                try:
                    final_ent = dict(entity_content)
                except:
                    pass

            # 补充唯一 ID（优先用内部 chunk_id，没有则补外层 id）
            if "id" not in final_ent and "chunk_id" not in final_ent:
                final_ent["id"] = doc.id

            # 补充相似度分数
            if hasattr(doc, "distance"):
                final_ent["score"] = doc.distance

        # ==============================================
        # 情况B：doc 已经是字典（模拟数据 / 已格式化数据）
        # ==============================================
        elif isinstance(doc, dict):
            # 子情况：字典嵌套 entity 结构 {entity:{...}, id:...}
            if "entity" in doc:
                ent = doc["entity"]
                if isinstance(ent, dict):
                    final_ent = ent.copy()
                # 补充 ID 和分数
                if "id" in doc and "id" not in final_ent:
                    final_ent["id"] = doc["id"]
                if "distance" in doc:
                    final_ent["score"] = doc["distance"]
            else:
                # 扁平字典，直接使用
                final_ent = doc

        # ==============================================
        # 情况C：支持 .get() 方法的其他对象
        # ==============================================
        elif hasattr(doc, "get"):
            ent = doc.get("entity") or doc
            if isinstance(ent, dict):
                final_ent = ent

        # 只保留合法非空字典
        if final_ent and isinstance(final_ent, dict):
            out.append(final_ent)

    return out

# ================================
# RRF 核心算法：倒数排序融合
# 作用：把多路召回结果按排名加权融合，自动去重、重新排序
# ================================
def reciprocal_rank_fusion(
        source_weights: list,
        k: int = 60,
        max_results: int = None,
) -> List[tuple]:

    # 存储每个文档的总得分
    score_map = {}
    # 存储每个文档完整内容
    chunk_map = {}

    # ==============================================
    # 遍历每一路召回结果，计算 RRF 分数
    # ==============================================
    for docs, weight in source_weights:
        # rank 从 1 开始（第一名=1，第二名=2...）
        for rank, item in enumerate(docs, start=1):
            # 获取文档唯一标识（chunk_id 优先，否则用 id）
            chunk_id = item.get("chunk_id") or item.get("id")

            if not chunk_id:
                logger.warning(
                    f"RRF Warning: item missing chunk_id/id: {list(item.keys()) if isinstance(item, dict) else item}")
                continue

            # ====================
            # RRF 公式核心
            # score += 权重 * (1 / (k + rank))
            # ====================
            score_map[chunk_id] = score_map.get(chunk_id, 0.0) + weight * (1.0 / (k + rank))

            # 只保存第一次出现的文档（去重）
            chunk_map.setdefault(chunk_id, item)
    # ==============================================
    # 按 RRF 总分排序
    # ==============================================
    merged = []
    for chunk_id, score in score_map.items():
        doc_item = chunk_map[chunk_id]
        merged.append((doc_item, score))

    # 得分从高到低排序
    merged.sort(key=lambda x: x[1], reverse=True)

    # 截断最多返回 N 条
    if max_results is not None:
        merged = merged[:max_results]

    return merged

# ================================
# 本地测试入口
# ================================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rrf 本地测试")
    print("=" * 50)

    mock_state = {
        "session_id": "test_rrf_session",
        "is_stream": False,
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
        "item_names": ["HAK 180 烫金机"]
    }

    try:
        from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
        from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde

        emb_res = node_search_embedding(mock_state)
        hyde_res = node_search_embedding_hyde(mock_state)
        mock_state['embedding_chunks'] = emb_res.get("embedding_chunks") or []
        mock_state['hyde_embedding_chunks'] = hyde_res.get("hyde_embedding_chunks") or []

        result = node_rrf(mock_state)
        rrf_chunks = result.get("rrf_chunks", [])

        emb_cnt = len(mock_state.get("embedding_chunks") or [])
        hyde_cnt = len(mock_state.get("hyde_embedding_chunks") or [])

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"输入数量: Embedding={emb_cnt}, HyDE={hyde_cnt}")
        print(f"输出数量: {len(rrf_chunks)}")
        print("-" * 30)

        print("最终排名:")
        for i, doc in enumerate(rrf_chunks, 1):
            doc_id = doc.get("chunk_id") or doc.get("id")
            content = (doc.get("content") or "")[:20]
            print(f"Rank {i}: ID={doc_id}, Content={content}...")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")


