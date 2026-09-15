import sys
import os
from app.utils.task_utils import add_running_task,add_done_task
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import create_hybrid_search_requests,hybrid_search,get_milvus_client
from app.core.logger import logger
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

def node_search_embedding(state):
    """
    【听书领域】向量检索节点：item_name 过滤 + 多字段组合过滤
    """
    logger.info("---search_embedding 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])

    # 1. 从会话状态中提取核心入参（听书新字段优先）
    query = state.get("rewritten_query") or state.get("original_query")
    item_names = state.get("item_names") or state.get("item_names") or []
    author_filter = state.get("author_filter", "")
    category_filter = state.get("category_filter", "")
    content_type_filter = state.get("content_type_filter", "")

    logger.info(f"核心入参提取：query='{query}', item_names={item_names}, author_filter={author_filter}")

    # 2. 对改写后的用户问题执行向量化，生成BGEM3稠密+稀疏向量
    logger.info(f"开始为文本获取嵌入值：{query[:50]}..." if len(query)>50 else f"开始为{query}文本获取嵌入值...")
    embeddings = generate_embeddings([query])

    dense_vec = embeddings.get("dense")[0]
    sparse_vec = embeddings.get("sparse")[0]
    logger.debug(f"向量生成成功：dense_dim={len(dense_vec)}, sparse_len={len(sparse_vec)}")

    # 3. 准备Milvus集合名：优先听书集合，兼容 .env 拼写差异
    from app.conf.milvus_config import milvus_config
    collection_name = (milvus_config.audiobook_chunks_collection
                       or milvus_config.chunks_collection
                       or os.environ.get("CHUNKS_COLLECTION")
                       or os.environ.get("CHUNK_COLLECTION"))
    logger.info(f"正在连接到 Milvus 并准备集合 '{collection_name}'...")

    # 4. 构造Milvus混合搜索请求对象（听书领域多字段组合过滤）
    expr_parts = []
    if item_names:
        quoted = ", ".join(f'"{v}"' for v in item_names)
        expr_parts.append(f"item_name in [{quoted}]")
    if author_filter:
        expr_parts.append(author_filter)
    if category_filter:
        expr_parts.append(category_filter)
    if content_type_filter:
        expr_parts.append(content_type_filter)

    expr = " and ".join(expr_parts) if expr_parts else None
    logger.info(f"Milvus 过滤表达式: {expr or '(无)'}")

    # 构造稠密+稀疏混合搜索请求
    reqs = create_hybrid_search_requests(
        dense_vector=dense_vec,
        sparse_vector=sparse_vec,
        expr=expr,
        limit=10
    )

    # 5. 执行Milvus稠密+稀疏混合向量检索（核心调用）
    logger.info("开始执行 Milvus 混合检索...")
    client = get_milvus_client()
    res = hybrid_search(
        client=client,
        collection_name=collection_name,
        reqs=reqs,
        ranker_weights=(0.8, 0.2),
        norm_score=True,
        limit=5,
        output_fields=["chunk_id", "content", "item_name", "author_name",
                       "content_type", "category", "file_title"]
    )

    # 打印节点处理成功日志
    hit_count = len(res[0]) if res and len(res)>0 else 0
    logger.info(f"节点 search_embedding 处理成功，检索到 {hit_count} 条相关片段")
    if hit_count>0:
        logger.debug(f"Top1 检索结果示例：{res[0][0]}")

    # 标记当前任务完成
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 6. 构造并返回结果
    return {"embedding_chunks": res[0] if res else []}


if __name__ == "__main__":
    # 模拟测试数据
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "三体的主要内容",  # 模拟改写后的查询
        "item_names": ["三体"],  # 模拟已确认的商品名
        "is_stream": False
    }

    print("\n>>> 开始测试 node_search_embedding 节点...")
    try:
        # 执行节点函数
        result = node_search_embedding(test_state)
        logger.info(f"检索结果汇总：{result}")
        # 验证结果
        chunks = result.get("embedding_chunks", [])
        print(f"\n>>> 测试完成！检索到 {len(chunks)} 条结果")

        if chunks:
            print("\n>>> Top 1 结果详情:")
            top1 = chunks[0]
            # 打印关键字段（注意：entity字段可能包含具体业务数据）
            print(f"ID: {top1.get('id')}")
            print(f"Distance: {top1.get('distance')}")
            entity = top1.get('entity', {})
            print(f"Item Name: {entity.get('item_name')}")
            print(f"Content Preview: {entity.get('content', '')[:100]}...")
        else:
            print("\n>>> 警告：未检索到任何结果，请检查 Milvus 数据或 item_names 是否匹配")

    except Exception as e:
        logger.error(f"测试运行失败: {e}", exc_info=True)