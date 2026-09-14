import os
from typing import Tuple

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import DataType

from app.conf.milvus_config import milvus_config
# 导入自定义模块：
# 1. 流程状态载体：ImportGraphState为LangGraph流程的统一状态管理对象
from app.import_process.agent.state import ImportGraphState
# 2. Milvus工具：获取单例Milvus客户端，实现连接复用
from app.clients.milvus_utils import get_milvus_client
# 3. 大模型工具：获取大模型客户端，统一模型调用入口
from app.lm.lm_utils import get_llm_client
# 4. 向量工具：BGE-M3模型实例、向量生成方法（稠密+稀疏向量）
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
# 5. 稀疏向量工具：归一化处理，保证向量长度为1，提升检索准确性
from app.utils.normalize_sparse_vector import normalize_sparse_vector
# 6. 任务工具：更新任务运行状态，用于任务监控和管理
from app.utils.task_utils import add_running_task, add_done_task
# 7. 日志工具：项目统一日志入口，分级输出（info/warning/error）
from app.core.logger import logger,node_log,step_log
# 8. 提示词工具：加载本地prompt模板，实现提示词与代码解耦
from app.core.load_prompt import load_prompt

from app.utils.escape_milvus_string_utils import escape_milvus_string

# --- 配置参数 (Configuration) ---
# 大模型识别商品名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500

from app.utils.escape_milvus_string_utils import escape_milvus_string

"""
  主要目标：
     1. 录用文本大模型识别当前chunks对应的item_name！用于区分不同的文档
     2. 使用嵌入式模型，将item_name生成向量存储到向量数据库 
     3. 修改state[chunks] -> chunk {title parent_title part file_title content item_name => 每个赋值 }
  实现步骤：
     1. 校验和取值 （file_title,chunks）
     2. 构建上下文环境  chunks -> top 5 -> 拼接成context文本 
     3. 调用模型，拼接提示词，识别chunks对应item_name
     4. 修改state chunks -》 item_name 
     5. item_name生成向量（稠密/稀疏）
     6. 存储向量到向量数据库 kb_item_name (id / file_title / item_name / 稠密 和 稀疏)
 """

from app.core.logger import logger, node_log
from app.import_process.agent.state import ImportGraphState


@node_log("node_item_name_recognition")
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    【听书领域改造】
    原：LLM 识别 item_name（商品名）
    新：LLM 识别 item_name（书名）+ author_name（作者）+ content_type（内容类型）
    """
    add_running_task(state['task_id'], 'node_item_name_recognition')

    chunks, file_title = step_1_get_chunks_and_file_title(state)
    context = step_2_build_context(chunks)

    # 3. 识别书籍信息：上传时已知书名则直接沿用，否则调用 LLM 识别
    known_item_name = state.get('item_name', '')
    known_author = state.get('author_name', '')
    known_ct = state.get('content_type', '')

    if known_item_name:
        # 用户已在 /upload 传入书名/作者/内容类型，无需 LLM 识别
        book_info = {
            "item_name": known_item_name,
            "author_name": known_author or "",
            "content_type": known_ct or "",
        }
        logger.info(f"使用上传元数据: item_name={known_item_name}, author={known_author}, ct={known_ct}")
    else:
        book_info = step_3_call_llm_for_book(context, file_title)

    item_name = book_info.get("item_name", "") or file_title
    author_name = book_info.get("author_name", "") or known_author or ""
    content_type = book_info.get("content_type", "") or known_ct or ""

    # 书名清洗：去掉 LLM/文件名 混入的内容类型后缀，如「三体简介」→「三体」
    item_name = _clean_item_name(item_name)

    # 4. 更新 state + chunks（同时写新老字段）
    step_4_update_chunks_and_state_v2(state, item_name, author_name, content_type, chunks)

    # 5. item_name 生成向量
    dense_vector, sparse_vector = step_5_generate_embeddings(item_name)

    # 6. 保存到书名集合（含元数据）
    step_6_save_book_vector_db(file_title, item_name, author_name, content_type, dense_vector, sparse_vector)

    add_done_task(state['task_id'], 'node_item_name_recognition')
    return state


@step_log("step_1_get_chunks_and_file_title")
def step_1_get_chunks_and_file_title(state) -> Tuple[str, str]:
    """
    继续进行参数校验和处理!
    :param state:
    :return:
    """
    chunks = state.get('chunks')
    file_title = state.get('file_title')

    if not chunks:
        raise ValueError("chunks没有值，无法继续进行，抛出异常处理！")
    if not file_title:
        # file_title没有值！
        # md_path中获取文件名即可 (字符串处理更方便)
        file_title = os.path.basename(state.get('md_path'))
        state['file_title'] = file_title
    return chunks, file_title


@step_log("step_2_build_context")
def step_2_build_context(chunks) -> str:
    """
    构建提示词上下文环境
    根据chunks切面的content内容进行分拼接！ （2000）
    截取内容限制： 1. 最多截取前top个 （5） 2. 最多字符不能超过 CONTEXT_TOTAL_MAX_CHARS
    截取内容处理：
          切片：{1}，标题:{title},内容：{content} \n\n
          切片：{2}，标题:{title},内容：{content} \n\n
          切片：{3}，标题:{title},内容：{content} \n\n
          切片：{4}，标题:{title},内容：{content} \n\n
          切片：{5}，标题:{title},内容：{content} \n\n
    :param chunks:
    :return:
    """
    # 1. 前置准备工作
    parts = []  # 存储处理后的切片：{1}，标题:{title},内容：{content} \n\n
    total_chars = 0  # 记录已经加入列表的字符串数量
    # 2. 循环处理 content + 判断
    for index, chunk in enumerate(chunks[:DEFAULT_ITEM_NAME_CHUNK_K], start=1):
        chunk_title = chunk['title']
        chunk_content = chunk['content']
        # 先处理一下！！
        # if len(chunk_content) + total_chars > SINGLE_CHUNK_CONTENT_MAX_LEN:
        #     chunk_content = chunk_content[:SINGLE_CHUNK_CONTENT_MAX_LEN-total_chars]
        data = f"切片：{index}，标题:{chunk_title},内容：{chunk_content}"
        parts.append(data)
        total_chars += len(data)
        # 第一次的content已经超标了但是完成了拼接！！！
        if total_chars >= CONTEXT_TOTAL_MAX_CHARS:
            logger.info(f"已经达到最大字符数:{total_chars}，停止拼接！")
            break
    # 结果的转化
    context = "\n\n".join(parts)
    # 兜底处理下
    final_context = context[:SINGLE_CHUNK_CONTENT_MAX_LEN]
    # 返回结果
    return final_context


@step_log("step_3_call_llm")
def step_3_call_llm(context, file_title) -> str:
    """[兼容旧入口] 调用 LLM 获取书名，兜底返回 file_title"""
    book_info = step_3_call_llm_for_book(context, file_title)
    return book_info.get("item_name", "") or file_title


@step_log("step_3_call_llm_for_book")
def step_3_call_llm_for_book(context, file_title) -> dict:
    """
    【听书领域】调用 LLM 识别书籍完整信息
    返回 dict: {"item_name": str, "author_name": str, "content_type": str}
    """
    import json as _json

    # 1. 构建提示词
    human_prompt = load_prompt("item_name_recognition", file_title=file_title, context=context)
    system_prompt = load_prompt("product_recognition_system")
    messages = [
        HumanMessage(content=human_prompt),
        SystemMessage(content=system_prompt)
    ]

    # 2. 获取模型（启用 json_mode 以稳定解析）
    llm = get_llm_client(json_mode=True)
    chain = llm | StrOutputParser()

    default_result = {"item_name": file_title, "author_name": "", "content_type": ""}

    try:
        raw = chain.invoke(messages)
        cleaned = raw
        # 兼容 ```json``` 包裹
        if cleaned.startswith("```json"):
            cleaned = cleaned.replace("```json", "").replace("```", "")
        result = _json.loads(cleaned)
        logger.info(f"LLM 识别结果: {result}")

        if not result.get("item_name"):
            result["item_name"] = file_title
        if not result.get("author_name"):
            result["author_name"] = ""
        if not result.get("content_type"):
            result["content_type"] = ""
        return result
    except Exception as e:
        logger.warning(f"LLM 结构化解析失败 ({e})，使用 file_title 兜底")
        return default_result


@step_log("step_4_update_chunks_and_state_v2")
def step_4_update_chunks_and_state_v2(state, item_name, author_name, content_type, chunks):
    """
    【听书领域】state + chunks 都写入听书元数据
    """
    state['item_name'] = item_name
    state['author_name'] = author_name
    state['content_type'] = content_type

    # 额外元数据（上传时已知，从 state 兜底读取并下沉到每个切片）
    category = state.get('category', '')
    audio_duration = state.get('audio_duration', '')
    source_path = state.get('source_path', '')
    narrator = state.get('narrator', '')

    for chunk in chunks:
        chunk['item_name'] = item_name
        chunk['author_name'] = author_name
        chunk['content_type'] = content_type
        chunk['category'] = category
        chunk['audio_duration'] = audio_duration
        chunk['source_path'] = source_path
        chunk['narrator'] = narrator
    state['chunks'] = chunks
    logger.info(f"听书元数据: item_name={item_name}, author={author_name}, ct={content_type}")


@step_log("step_5_generate_embeddings")
def step_5_generate_embeddings(item_name):
    """
    根据item_name生成向量 -》 稠密 + 稀疏
    :param item_name:
    :return: dense_vector [稠密] ,  sparse_vector [稀疏]
    """
    """
    generate_embeddings 自己封装的嵌入式模式生成向量的函数！！ 
          embeddings list对应的向量 = model.encode_documents(texts) 传入的字符串list 
          参数：生成向量的字符串 ["1","2","3"] 
          返回结果： 
             result = {
                        "dense": [1的稠密,2的稠密,3的稠密],  #稠密向量
                        "sparse": [1的稀疏,2的稀疏,3的稀疏], #稀疏向量
                      }
    """
    result = generate_embeddings([item_name])
    dense_vector, sparse_vector = result['dense'][0], result['sparse'][0]
    return dense_vector, sparse_vector


# 书名清洗：常见内容类型/资料类型后缀，识别终点名时将其裁掉
_BOOK_NAME_SUFFIXES = [
    "书籍简介", "作者介绍", "作者简介", "听书笔记", "有声书信息", "有声书",
    "推荐运营资料", "推荐语", "编辑推荐", "用户评论摘要", "评论摘要", "常见问答",
    "简介", "介绍", "笔记", "书评", "评论", "全文", "正文", "目录", "大纲",
]
# 匹配后缀（可带 . 或 _ 或空格 分隔，如「三体_简介」「三体.简介」）
_BOOK_NAME_SUFFIX_RE = None


def _get_suffix_re():
    global _BOOK_NAME_SUFFIX_RE
    if _BOOK_NAME_SUFFIX_RE is None:
        import re as _re
        pattern = r"(?:[._\-\s]*)(?:" + "|".join(sorted(_BOOK_NAME_SUFFIXES, key=len, reverse=True)) + r")\s*$"
        _BOOK_NAME_SUFFIX_RE = _re.compile(pattern)
    return _BOOK_NAME_SUFFIX_RE


def _clean_item_name(item_name: str) -> str:
    """
    去除书名中混入的内容类型后缀（如「三体简介」→「三体」）。
    后缀剥除时书名不应被削减为空，否则保留原名。
    """
    if not item_name:
        return item_name
    name = str(item_name).strip()
    cleaned = _get_suffix_re().sub("", name).strip()
    return cleaned if cleaned else name


@step_log("step_6_save_book_vector_db")
def step_6_save_book_vector_db(file_title, item_name, author_name, content_type, dense_vector, sparse_vector):
    """
    【听书领域】保存书名向量+元数据到 Milvus 书名集合
    """
    collection_name = milvus_config.item_name_collection
    milvus_client = get_milvus_client()

    if not milvus_client.has_collection(collection_name=collection_name):
        schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="author_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="content_type", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_params = milvus_client.prepare_index_params()
        index_params.add_index(field_name="dense_vector", index_name="dense_vector_index",
                               index_type="HNSW", metric_type="COSINE",
                               params={"M": 16, "efConstruction": 200})
        index_params.add_index(field_name="sparse_vector", index_name="sparse_vector_index",
                               index_type="SPARSE_INVERTED_INDEX", metric_type="IP",
                               params={"inverted_index_algo": "DAAT_MAXSCORE"})
        milvus_client.create_collection(collection_name=collection_name,
                                        schema=schema, index_params=index_params)

    milvus_client.load_collection(collection_name=collection_name)
    safe_key = escape_milvus_string(item_name)
    milvus_client.delete(collection_name=collection_name, filter=f"item_name=='{safe_key}'")

    item = {
        "file_title": file_title,
        "item_name": item_name,
        "author_name": author_name,
        "content_type": content_type,
        "dense_vector": dense_vector,
        "sparse_vector": sparse_vector
    }
    milvus_client.insert(collection_name=collection_name, data=[item])
    milvus_client.load_collection(collection_name=collection_name)
    logger.info(f"保存书名向量: item_name={item_name}, author={author_name}, ct={content_type}")


# ===================== 本地测试方法（直接运行调试，无需启动LangGraph） =====================
def test_node_item_name_recognition():
    """
    【听书领域】书籍信息识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_item_name_recognition节点全链路逻辑
    """
    logger.info("=== 开始执行【听书书籍信息识别】节点本地测试 ===")
    try:
        # 1. 构造模拟的 ImportGraphState（听书场景测试数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",
            "file_title": "三体_书籍简介",
            "chunks": [
                {
                    "title": "内容简介",
                    "content": "《三体》是刘慈欣创作的系列长篇科幻小说，由《三体》《三体Ⅱ·黑暗森林》《三体Ⅲ·死神永生》组成。作品讲述了地球人类文明和三体文明的信息交流、生死搏杀及两个文明在宇宙中的兴衰历程。"
                },
                {
                    "title": "作者简介",
                    "content": "刘慈欣，1963年6月出生于北京，山西阳泉人，科幻作家，被誉为中国科幻文学的里程碑，凭借《三体》获第73届世界科幻大会颁发的雨果奖最佳长篇小说奖，为亚洲首次获奖。"
                },
                {
                    "title": "有声书信息",
                    "content": "《三体》有声书由王明军演播，总时长约34小时，共120集，语速适中，适合夜间收听。"
                }
            ]
        })

        # 2. 调用听书书籍识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印听书领域测试结果
        logger.info("=== 听书书籍信息识别节点测试完成 ===")
        logger.info(f"测试任务ID: {result_state.get('task_id')}")
        logger.info(f"识别书名(item_name): {result_state.get('item_name')}")
        logger.info(f"识别作者(author_name): {result_state.get('author_name')}")
        logger.info(f"识别内容类型(content_type): {result_state.get('content_type')}")
        logger.info(f"切片数量: {len(result_state.get('chunks', []))}")

    except Exception as e:
        logger.error(f"听书书籍信息识别节点测试失败，原因: {str(e)}", exc_info=True)


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_item_name_recognition()
