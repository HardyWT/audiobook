import json
import os
import re
from pathlib import Path
from typing import Tuple, List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task

# ====================== 全局配置（可根据模型调整）======================
# 单个文本块最大长度（控制不超过模型上下文）
CHUNK_SIZE = 200 # 小值方便测试切割
# 块之间重叠长度（保证语义不丢失）
CHUNK_OVERLAP = 20

"""
1. **获取与清洗内容 (Step 1)**
   从 `state` 中提取 Markdown 内容与文件标题，统一换行符格式（`\r\n` / `\r` → `\n`），保证跨平台兼容。
2. **按标题语义初切 (Step 2)**
   基于 Markdown 标题语法（`#` ~ `######`）进行**语义级切分**，自动跳过代码块内的标题匹配，避免误切注释，保证每个块语义完整。
3. **无标题文档兜底处理 (Step 2 内置)**
   若文档无任何标题，自动生成默认标题 `无主题`，确保内容不丢失、流程不中断。
4. **超长块递归精细化切割 (Step 3)**
   使用 `RecursiveCharacterTextSplitter` 对**超过指定长度**的语义块进行二次切割，按「段落 → 换行 → 句子 → 空格」优先级切割，**不产生碎片、不硬断句子、无需手动合并**。
5. **构建标准 Chunk 结构**
   为每个切片补充完整元数据：`title`、`content`、`file_title`、`parent_title`、`part` 序号，保证可检索、可溯源。
6. **本地备份与状态更新 (Step 4)**
   将切分结果备份到本地 `chunks.json` 文件，同时将最终 chunks 存入 `state`，供后续向量入库使用。
"""

@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    未来要实现:
    1. 基于 Markdown 标题层级进行递归切分。
    2. 对过长的段落进行二次切分。
    3. 生成包含 Metadata (标题路径) 的 Chunk 列表。
    """

    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    """
    # 1. 进行任务和日志处理
    add_running_task(state['task_id'],'node_document_split')
    # 2. 进行state中数据清晰(md_content / file_title (做标题兜底))
    md_content, file_title = step_1_get_content(state)
    # 3. 按标题语义初切
    # [{content:标题的内容,title：标题,file_title：文件名},{},{}]
    sections,title_count,lines_count = step_2_split_by_title(md_content,file_title)
    # 4. 进行语义内递归切割
    # [{content:标题的内容,title：标题,file_title：文件名,parent_title,part},{},{}]
    final_chunks = step_3_refine_chunks(sections)
    # 5. 数据备份和修改state chunks
    state['chunks'] = final_chunks
    step_4_backup_chunks(final_chunks,state)
    add_done_task(state['task_id'], 'node_document_split')
    return state

@step_log("step_1_get_content")
def step_1_get_content(state) -> Tuple[str, str]:
    """
    数据清晰,处理md_content中不同系统的换成分割! 统一处理
    并且获取文件file_tile用于整个内容title兜底
    :param state:
    :return:
    """
    # 1. 获取md_content内容
    md_content = state['md_content']
    if not md_content:
        logger.error(f"没有输入内容,请检查输入内容是否正确!")
        raise RuntimeError("没有输入内容,请检查输入内容是否正确!")
    # 2.清晰数据统一换行符号
    """
        window \r\n
        linux/mac \n
        老mac   \r
    """
    md_content = md_content.replace('\r\n', '\n').replace('\r', '\n')
    file_title = state.get("file_title", "default_file")
    return md_content, file_title


@step_log("step_2_split_by_title")
def step_2_split_by_title(md_content, file_title) -> List[Dict]:
    """
     语义切割,根据标题,进行内容切割!
    :param md_content:
    :param file_title:
    :return: [{content,title,file_title}]
    """
    # 1. 定义切割正则 / md_content按行切割
    # \s* 空格 tab * 0 - n
    # #{1,6} 匹配1-6个 #
    # \s+  + 1->n   #### 标题名
    # .+ .任意字符串 + 1->n   [空格]###[空格]标题描述
    title_pattern = re.compile(r'^\s*#{1,6}\s+.+')
    lines = md_content.split('\n')
    # 准备存储数据容器
    chunks = []  # 最终结果
    current_title = ""  # 当前标题
    current_lines = []  # 当前标题还行内容
    in_code_block = False  # 记录是否在代码块中
    title_count = 0

    # 2. 循环处理每行数据
    for line in lines:
        strip_line = line.strip()
        # 判断是否在代码块中
        if strip_line.startswith("```") or strip_line.startswith("~~~"):
            in_code_block = not in_code_block  # 取反即可
            current_lines.append(line)
            continue
        # 不是代码块,判断是不是标题
        if not in_code_block and title_pattern.match(strip_line):
            # 到了新的标题,将上一次标题进行除虫脲
            if current_title:
                # 存储上一次标题
                chunks.append({
                    "title": current_title,
                    "content": "\n".join(current_lines),
                    "file_title": file_title
                })
            # 重置变量(记录当前行)
            current_title = strip_line
            current_lines = [strip_line]
            title_count += 1
        else:
            # 添加行内容 [还是标题内容,非代码行]
            current_lines.append(strip_line)
    # 3. 最后一块存储 (最后一次跳出循环没有保存)
    if current_title:
        chunks.append({
            "title": current_title.strip(),
            "content": "\n".join(current_lines),
            "file_title": file_title
        })
    # 4. 进行无标题处理
    # --------------------
    # 兜底：文档无标题时
    # --------------------
    if not chunks:
        chunks = [{
            "title": "无主题",
            "content": md_content,
            "file_title": file_title
        }]

    """
    md -> ##  # - ######[空格]标题名称

    ## 开篇
    内容 \n
    ![]()
    ```  ~~~python 代码块
       # 注释
       # 注释
       python 
    内容 \n

    ## 中篇
    内容 \n
    xxxxx
    内容 \n

    ##  下篇
    内容 \n
    内容 \n 
    """
    return chunks, title_count, len(lines)

@step_log("step_3_refine_chunks")
def step_3_refine_chunks(sections) -> List[Dict]:
    """
      同一标题下,同一语义,进行二次超长切割!!
    :param sections: 按标题切割数据
    :return: 二次切割数据
    """
    spliter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # 切割优先级：段落 → 换行 → 句子 → 空格
        separators=["\n\n", "\n", "。", "！", "；", " "]
    )
    # 进行切割
    final_chunks = []
    for section in sections:
        # 进行二次切割
        sub_chunks = spliter.split_text(section["content"])
        has_multiple_chunks = len(sub_chunks) > 1
        # 生成带编号的子块
        for idx,chunk in enumerate(sub_chunks,start=1):
            current_title = f"{section['title']}_{idx}" if has_multiple_chunks else section["title"]
            final_chunks.append({
                "title": current_title,
                "content": chunk.strip(),
                "file_title": section["file_title"],
                "parent_title": section["title"],
                "part": idx
            })
    return final_chunks

def step_4_backup_chunks(final_chunks, state):
    """
      进行最终数据备份
    :param final_chunks: 要备份的数据
    :param state: 获取local_dir文件夹
    :return:
    """
    backup_file_path = Path(state['md_path']).parent / "backup_chunks.json"

    with open(backup_file_path, "w", encoding="utf-8") as f:
        json.dump(
            final_chunks,
            f,
            ensure_ascii=False, #中文直接原文存储
            indent=4 # json带有缩进 4
        )
    logger.debug(f"数据备份成功,备份地址:{backup_file_path}")

if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "hak180产品安全手册",
            "local_dir":os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")