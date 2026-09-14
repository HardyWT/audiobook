from typing import TypedDict, Optional
import copy
from app.core.logger import logger

# ====================== 听书领域常量 ======================
# 内容类型枚举（需求文档第5节定义的7种内容类型）
CONTENT_TYPES = [
    "有声书信息", "书籍简介", "作者介绍", "听书笔记",
    "推荐运营资料", "用户评论摘要", "常见问答"
]


class ImportGraphState(TypedDict, total=False):
    """
    听书知识库 - 导入流程图状态定义
    领域变更：硬件/电子产品(item_name) → 听书/有声书(item_name)
    新增元数据：author_name / content_type / category / audio_duration
    """
    task_id: str
    is_md_read_enabled: bool
    is_pdf_read_enabled: bool
    local_dir: str
    local_file_path: str
    file_title: str
    pdf_path: str
    md_path: str
    source_path: str

    # --- 听书领域核心元数据（需求第5节）---
    item_name: str
    author_name: str
    content_type: str
    category: str
    audio_duration: str
    narrator: str

    md_content: str
    chunks: list
    embeddings_content: list


graph_default_state: ImportGraphState = {
    "task_id": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "source_path": "",
    # 听书领域元数据
    "item_name": "",
    "author_name": "",
    "content_type": "",
    "category": "",
    "audio_duration": "",
    "narrator": "",
    "md_content": "",
    "chunks": [],
    "embeddings_content": []
}


def create_default_state(**overrides) -> ImportGraphState:
    """创建默认状态并支持覆盖字段（保留项目原入口，兼容已有调用）"""
    state = copy.deepcopy(graph_default_state)
    state.update(overrides)
    return state


def get_default_state() -> ImportGraphState:
    """获取默认状态的副本，避免多任务共享同一份可变对象"""
    return copy.deepcopy(graph_default_state)


if __name__ == "__main__":
    # 本地测试
    s = create_default_state(task_id="task_001", local_file_path="三体_简介.md")
    print(s["task_id"], s["local_file_path"], s["item_name"])