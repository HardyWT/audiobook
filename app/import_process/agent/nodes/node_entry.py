from pathlib import Path
from app.core.logger import logger, node_log
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.task_utils import add_running_task, add_done_task

@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    未来要实现:
    1. 接收文件路径。
    2. 判断文件类型 (PDF/MD)。
    3. 设置 state 中的路由标记 (is_pdf_read_enabled / is_md_read_enabled)。
    """

    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    未来要实现:
        1. 进行任务状态记录,开始和结束列表记录
        2. 根据state中 local_file_path属性判断数据类型进而修改
           相关参数, is_md_read_enabled 或者 is_pdf_read_enabled
                    md_path 或者 pdf_path
        3. 不可解析结果类型不可用,直接输出对应警告日志! 逻辑路由节点会自动处理
        4. 获取file_tile标识,用于后期识别pdf对应的主体(item_name)进行兜底
    """

    # 1. 任务状态记录处理
    add_running_task(state['task_id'],'node_entry')

    # 2. 判断文件类型
    local_file_path = state['local_file_path']
    if not local_file_path:
        logger.warning(f"没有输入文件地址,无法处理,直接跳转到结束节点!")
        add_done_task(state['task_id'], 'node_entry')
        return state

    if local_file_path.endswith(".md"):
        state['is_md_read_enabled'] = True
        state['md_path'] = local_file_path
    elif local_file_path.endswith(".pdf"):
        state['is_pdf_read_enabled'] = True
        state['pdf_path'] = local_file_path
    else:
        logger.warning(f"虽然输出了local_file_path,但是无法识别文件类型,请检查输入文件类型是否正确,目前只支持md和pdf文件,请检查! {local_file_path}")
        add_done_task(state['task_id'], 'node_entry')
        return state

    # 3. 获取文件标识
    file_title = Path(local_file_path).stem # 文件名 .name  文件夹名 .parent   文件后缀 .suffix
    state['file_title']= file_title

    # ===== 听书领域：元数据铺垫 =====
    # 注意：这里【不要】把 file_title 直接写进 item_name！
    #   否则文件名「三体简介.md」会被当成书名，导致后续 LLM 无法识别出真正的书名「三体」。
    #   书名由 node_item_name_recognition 负责识别（上传时显式传入的 item_name 除外）。
    logger.info(
        f"听书元数据: item_name={state.get('item_name')}, "
        f"author={state.get('author_name')}, content_type={state.get('content_type')}, "
        f"category={state.get('category')}, audio_duration={state.get('audio_duration')}"
    )

    add_done_task(state['task_id'], 'node_entry')
    return state

if __name__ == '__main__':

    # 单元测试：覆盖不支持类型、MD、PDF三种场景
    logger.info("===== 开始node_entry节点单元测试 =====")

    # 测试1: 不支持的TXT文件
    test_state1 = create_default_state(
        task_id="test_task_001",
        local_file_path="三体简介.txt"
    )
    node_entry(test_state1)

    # 测试2: MD文件
    test_state2 = create_default_state(
        task_id="test_task_002",
        local_file_path="三体简介.md"
    )
    node_entry(test_state2)

    # 测试3: PDF文件
    test_state3 = create_default_state(
        task_id="test_task_003",
        local_file_path="三体简介.pdf"
    )
    node_entry(test_state3)

    logger.info("===== 结束node_entry节点单元测试 =====")