# HyDE节点
import sys
from app.utils.task_utils import add_running_task, add_done_task
from app.lm.lm_utils import *
from app.lm.embedding_utils import *
from app.clients.milvus_utils import *
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


def step_1_create_hyde_doc(rewritten_query: str) -> str:
    """
    阶段1：利用大模型根据用户查询生成假设性文档（Hypothetical Document）。
    HyDE的核心在于：利用LLM生成一个“虚构但相关”的文档，用该文档的向量去检索真实的文档，
    从而缓解短查询（Query）与长文档（Document）在语义空间不匹配的问题。

    :param rewritten_query: 用户改写后的查询语句
    :return: LLM生成的假设性文档内容
    """
    if not rewritten_query:
        logger.error("Step 1 Error: rewritten_query 为空")
        raise ValueError("rewritten_query 不能为空")

    logger.info(f"Step 1: 开始生成假设性文档 (HyDE), Query: {rewritten_query}")

    try:
        llm = get_llm_client()
        # 加载提示词模板，生成假设文档
        # 提示词通常引导LLM："请为这个问题写一段专业的回答..."
        hyde_prompt = load_prompt("hyde_prompt", rewritten_query=rewritten_query)
        logger.debug(f"Step 1: Prompt加载成功, 长度: {len(hyde_prompt)}")

        # 调用LLM生成
        response = llm.invoke(hyde_prompt)
        hyde_doc = response.content

        logger.info(f"Step 1: 假设文档生成完成, 长度: {len(hyde_doc)} 字符")
        logger.debug(f"Step 1: 文档预览: {hyde_doc[:50]}...")

        return hyde_doc

    except Exception as e:
        logger.error(f"Step 1: 生成假设文档失败: {e}")
        raise e


def step_2_search_embedding_hyde(
        rewritten_query: str,
        hyde_doc: str,
        item_names=None,
        author_filter=None,
        category_filter=None,
        content_type_filter=None,
        req_limit: int = 10,
        top_k: int = 5,
        ranker_weights=(0.8, 0.2),
        norm_score: bool = True,
        output_fields=["chunk_id", "content", "item_name", "author_name",
                       "content_type", "category", "file_title", "item_name"],
):
    """
    【听书领域】阶段2：利用"重写问题 + 假设性文档"，到向量库检索切片。

    :param rewritten_query: 改写后的查询
    :param hyde_doc: Step 1 生成的假设性文档
    :param item_names: 书名列表，用于元数据过滤 (item_name in [...])
    :param author_filter: 作者过滤 expr 片段
    :param category_filter: 类别过滤 expr 片段
    :param content_type_filter: 内容类型过滤 expr 片段
    :param req_limit: Milvus 搜索时的候选召回数量
    :param top_k: 最终返回的 Top K 结果数量
    :param ranker_weights: 混合检索权重 (Dense, Sparse)
    :param norm_score: 是否对分数进行归一化
    :param output_fields: 返回结果中包含的字段
    :return: 检索结果列表
    """
    if not rewritten_query:
        raise ValueError("rewritten_query 不能为空")
    if not hyde_doc:
        raise ValueError("hypothetical_doc 不能为空")

    # 1. 拼接查询与假设文档，形成更丰富的语义上下文
    combined_text = rewritten_query + " " + hyde_doc
    logger.info(f"Step 2: 拼接 Query + HyDE Doc, 总长度: {len(combined_text)}")

    # 2. 生成向量 (Dense + Sparse)
    logger.info("Step 2: 正在生成混合向量 (Embedding)...")
    embeddings = generate_embeddings([combined_text])

    # 3. Milvus 集合名：优先听书集合
    from app.conf.milvus_config import milvus_config
    collection_name = (milvus_config.audiobook_chunks_collection
                       or milvus_config.chunks_collection
                       or os.environ.get("CHUNKS_COLLECTION"))
    if not collection_name:
        logger.error("Step 2 Error: 未配置 chunks 集合名")
        return []

    logger.info(f"Step 2: 准备在集合 '{collection_name}' 中执行混合检索")

    # 构造过滤表达式（听书领域多字段组合）
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
    logger.info(f"Step 2: 过滤表达式: {expr or '(无)'}")

    try:
        # 构造搜索请求
        reqs = create_hybrid_search_requests(
            dense_vector=embeddings.get("dense")[0],
            sparse_vector=embeddings.get("sparse")[0],
            expr=expr,
            limit=req_limit,
        )

        client = get_milvus_client()
        if not client:
            logger.error("Step 2 Error: 无法连接到 Milvus")
            return []

        # 执行混合检索
        logger.info(f"Step 2: 执行 Hybrid Search, Weights={ranker_weights}, TopK={top_k}")
        res = hybrid_search(
            client=client,
            collection_name=collection_name,
            reqs=reqs,
            ranker_weights=ranker_weights,
            norm_score=norm_score,
            limit=top_k,
            output_fields=list(output_fields),
        )

        hit_count = len(res[0]) if res and len(res) > 0 else 0
        logger.info(f"Step 2: 检索完成, 找到 {hit_count} 个匹配切片")

        return res

    except Exception as e:
        logger.error(f"Step 2: 检索过程发生异常: {e}")
        return []

def node_search_embedding_hyde(state):
    logger.info("---HyDE （假设文档检索）节点开始处理---")
    # 记录任务开始状态
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state.get("is_stream"))

    # 1. 参数提取与校验
    # 优先使用改写后的查询，若无则降级使用原始查询
    rewritten_query = state.get("rewritten_query")
    if not rewritten_query:
        rewritten_query = state.get("original_query")

    if not rewritten_query:
        logger.error("HyDE节点错误: 未找到有效的用户查询 (rewritten_query/original_query 均为空)")
        return {}

    item_names = state.get("item_names") or state.get("item_names") or []
    author_filter = state.get("author_filter")
    category_filter = state.get("category_filter")
    content_type_filter = state.get("content_type_filter")
    logger.info(f"HyDE检索入参：query='{rewritten_query}',item_names={item_names}")

    # 阶段1：生成假设性文档
    try:
        logger.info("Step 1: 开始生成假设性文档 (HyDE Doc)...")
        hyde_doc = step_1_create_hyde_doc(rewritten_query)
        logger.info(f"Step 1: 假设文档生成成功 (长度: {len(hyde_doc)})")
        logger.debug(f"假设文档预览: {hyde_doc[:100]}...")
    except Exception as e:
        logger.error(f"Step 1 (生成假设文档) 发生异常: {e}", exc_info=True)
        # HyDE生成失败属于非阻断性错误，可选择直接返回空或降级处理，此处直接返回空结果
        return {}

    # 阶段2：用“重写问题 + 假设文档”检索切片
    try:
        logger.info("Step 2: 基于假设文档执行 Milvus 混合检索...")
        res = step_2_search_embedding_hyde(
            rewritten_query=rewritten_query,
            hyde_doc=hyde_doc,
            item_names=item_names,
            author_filter=author_filter,
            category_filter=category_filter,
            content_type_filter=content_type_filter,
            top_k=5,
        )

        hit_count = len(res[0]) if res and len(res) > 0 else 0
        logger.info(f"Step 2: 检索完成，召回 {hit_count} 条相关切片")

        if hit_count > 0:
            # 打印第一条结果用于调试
            first_hit = res[0][0]
            score = first_hit.get("distance")
            content_preview = first_hit.get("entity", {}).get("content", "")[:30]
            logger.debug(f"Top1 结果: Score={score}, Content='{content_preview}...'")

        return {
            "hyde_embedding_chunks": res[0] if res else [],
            "hyde_doc": hyde_doc,
        }
    except Exception as e:
        logger.error(f"Step 2 (向量生成与检索) 发生异常: {e}", exc_info=True)
        return {}
    finally:
        # 无论成功失败，均标记任务结束
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        logger.info("---HyDE 节点处理结束---")


if __name__ == "__main__":
    # 本地测试代码
    print("\n" + "=" * 50)
    print(">>> 启动 node_search_embedding_hyde 本地测试")
    print("=" * 50)

    # 模拟输入状态
    mock_state = {
        "session_id": "test_hyde_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
        "item_names": ["HAK 180 烫金机"],
        "is_stream": False
    }

    try:
        # 运行节点
        result = node_search_embedding_hyde(mock_state)

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"HyDE Doc Generated: {bool(result.get('hyde_doc'))}")
        if result.get("hyde_doc"):
            print(f"Doc Preview: {result.get('hyde_doc')[:50]}...")

        chunks = result.get("hyde_embedding_chunks", [])
        print(f"Chunks Found: {len(chunks)} , chunks内容：{chunks}")
        if chunks:
            print(f"Top Chunk Score: {chunks[0].get('distance')}")
        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")


