# 导入核心依赖（和其他配置类共用，只需导入一次）
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 提前加载.env配置文件（全局执行一次即可，无需重复写）
load_dotenv()

# ===================== 其他配置类（LLM/Embedding）可放在上方，保持原有代码不变 =====================
# ... 你的LLMConfig、EmbeddingConfig代码 ...

# 定义Milvus向量数据库配置类
@dataclass
class MilvusConfig:
    milvus_url: str  # Milvus服务端连接地址
    # 听书知识库集合名（优先使用）
    audiobook_chunks_collection: str  # 听书切片集合
    item_name_collection: str         # 听书-标题/实体集合
    # 兼容旧切片集合
    chunks_collection: str         # 存储切片的集合名称
    entity_name_collection: str    # 预留-实体名称集合

# 实例化Milvus配置对象（优先听书集合，未配置时降级旧集合）
milvus_config = MilvusConfig(
    milvus_url=os.getenv("MILVUS_URL"),
    audiobook_chunks_collection=os.getenv("AUDIOBOOK_CHUNKS_COLLECTION", os.getenv("CHUNKS_COLLECTION")),
    item_name_collection=os.getenv("ITEM_NAME_COLLECTION"),
    chunks_collection=os.getenv("CHUNKS_COLLECTION"),
    entity_name_collection=os.getenv("ENTITY_NAME_COLLECTION")
)