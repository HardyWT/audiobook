import shutil
from pathlib import Path
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, UploadFile, File, Form
from typing import List, Optional
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from app.core.logger import logger, PROJECT_ROOT
from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import get_default_state

from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
# from app.query_process.agent.main_graph import query_app

# 定义fastapi对象
app = FastAPI(title="query service", description="有声听书查询软件")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# 静态页面路由：返回文件导入前端页面import.html
# 访问地址：http://localhost:8000/import.html
# --------------------------
@app.get("/import.html", response_class=FileResponse)
async def get_import_page():
    """返回文件导入前端页面：import.html"""
    # 拼接HTML文件绝对路径，基于项目根目录定位
    html_abs_path = PROJECT_ROOT / "app/import_process/page/import.html"
    # 日志记录页面访问的文件路径，方便排查文件不存在问题
    logger.info(f"前端页面访问，文件绝对路径：{html_abs_path}")

    # 校验文件是否存在，不存在则抛出404异常
    if not os.path.exists(html_abs_path):
        logger.error(f"前端页面文件不存在，路径：{html_abs_path}")
        raise HTTPException(status_code=404, detail="import.html page not found")

    # 以FileResponse返回HTML文件，浏览器自动渲染
    return FileResponse(
        path=html_abs_path,
        media_type="text/html"  # 显式指定媒体类型为HTML，确保浏览器正确解析
    )

# --------------------------
# 后台任务：LangGraph全流程执行
# 独立于主请求线程，由BackgroundTasks触发，避免阻塞接口响应
# --------------------------
def run_graph_task(task_id: str, local_dir: str, local_file_path: str,
                   item_name: str = "", author_name: str = "",
                   content_type: str = "", category: str = "",
                   audio_duration: str = "", source_path: str = ""):
    """
    LangGraph全流程执行后台任务
    【听书领域】增加听书元数据参数，上传时已知的元数据直接注入 state
    """
    try:
        # 1. 更新任务全局状态为：处理中
        update_task_status(task_id, "processing")
        logger.info(f"[{task_id}] 开始执行LangGraph全流程，本地文件路径：{local_file_path}")

        # 2. 初始化LangGraph状态：加载默认状态 + 注入当前任务的核心参数
        init_state = get_default_state()
        init_state["task_id"] = task_id  # 任务ID关联
        init_state["local_dir"] = local_dir  # 任务本地目录
        init_state["local_file_path"] = local_file_path  # 上传文件本地路径

        # ===== 听书元数据注入 =====
        if item_name:
            init_state["item_name"] = item_name
        if author_name:
            init_state["author_name"] = author_name
        if content_type:
            init_state["content_type"] = content_type
        if category:
            init_state["category"] = category
        if audio_duration:
            init_state["audio_duration"] = audio_duration
        if source_path:
            init_state["source_path"] = source_path

        # 3. 流式执行LangGraph全流程（stream模式：实时获取每个节点的执行结果）
        for event in kb_import_app.stream(init_state):
            for node_name, node_result in event.items():
                # 记录每个节点完成的日志，包含任务ID和节点名，方便追踪执行顺序
                logger.info(f"[{task_id}] LangGraph节点执行完成：{node_name}")
                # 将完成的节点名加入【已完成列表】，前端轮询/status/{task_id}可实时获取
                add_done_task(task_id, node_name)

        # 4. 全流程执行完成，更新任务全局状态为：已完成
        update_task_status(task_id, "completed")
        logger.info(f"[{task_id}] LangGraph全流程执行完毕，任务完成")

    except Exception as e:
        # 5. 捕获全流程异常，更新任务全局状态为：失败，并记录错误日志（含堆栈）
        update_task_status(task_id, "failed")
        logger.error(f"[{task_id}] LangGraph全流程执行失败，异常信息：{str(e)}", exc_info=True)

# --------------------------
# 核心接口：多文件上传接口（不上传 MinIO）
# 支持多文件批量上传，核心流程：接收文件 → 本地保存 → 启动后台任务
# 访问地址：http://localhost:8000/upload （POST请求，form-data格式传参）
# --------------------------
from pathlib import Path

@app.post("/upload", summary="听书知识库文件上传", description="支持多文件批量上传，可附加听书元数据")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    item_name: str = Form("", description="书名（可选，优先使用，不传则由LLM自动识别）"),
    author_name: str = Form("", description="作者名（可选）"),
    content_type: str = Form("", description="内容类型：有声书信息/书籍简介/作者介绍/听书笔记/推荐运营资料/用户评论摘要/常见问答"),
    category: str = Form("", description="类别/标签：小说/非小说/儿童/教育/科幻/悬疑等"),
    audio_duration: str = Form("", description="有声书时长，如 '12小时35分钟'"),
    source_path: str = Form("", description="来源路径或资源链接"),
):
    """
    听书知识库文件上传核心接口（不上传 MinIO）
    1. 接收前端上传的多文件（PDF/MD为主）
    2. 按「日期/任务ID」分层保存到本地输出目录，避免文件冲突
    3. 为每个文件生成唯一TaskID，启动独立的LangGraph后台处理任务
    4. 实时更新任务状态，供前端轮询监控进度

    :param background_tasks: FastAPI后台任务对象，用于异步执行LangGraph流程
    :param files: 前端上传的文件列表（form-data格式）
    :param item_name/author_name/content_type/category/audio_duration/source_path: 听书元数据（可选）
    :return: 包含上传结果和所有任务ID的JSON响应
    """
    # 1. 构建本地存储根目录：项目根目录/output/YYYYMMDD（按日期分层，方便管理）
    today_str = datetime.now().strftime("%Y%m%d")
    date_based_root_dir: Path = PROJECT_ROOT / "output" / today_str

    # 初始化任务ID列表，用于返回给前端（一个文件对应一个TaskID）
    task_ids = []

    # 2. 遍历处理每个上传的文件（多文件批量处理，各自独立生成TaskID）
    for file in files:
        # 生成全局唯一TaskID（UUID4），作为单个文件的全流程标识
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        logger.info(f"[{task_id}] 开始处理上传文件，文件名：{file.filename}，文件类型：{file.content_type}")

        # 3. 标记「文件上传」阶段为「运行中」，前端轮询可查
        add_running_task(task_id, "upload_file")

        # 4. 构建该任务的本地独立目录：output/YYYYMMDD/TaskID，避免多文件重名冲突
        task_local_dir: Path = date_based_root_dir / task_id
        task_local_dir.mkdir(parents=True, exist_ok=True)

        # 5. 构建上传文件的本地保存绝对路径
        local_file_abs_path: Path = task_local_dir / file.filename

        # 6. 将上传的文件保存到本地临时目录
        with local_file_abs_path.open("wb") as file_buffer:
            shutil.copyfileobj(file.file, file_buffer)
        logger.info(f"[{task_id}] 文件已保存至本地，路径：{local_file_abs_path}")

        # 7. 标记「文件上传」阶段为「已完成」
        add_done_task(task_id, "upload_file")

        # 8. 将LangGraph全流程处理加入FastAPI后台任务（附带听书元数据）
        background_tasks.add_task(
            run_graph_task,
            task_id,
            str(task_local_dir),
            str(local_file_abs_path),
            item_name, author_name, content_type, category, audio_duration, source_path
        )
        logger.info(f"[{task_id}] 已将LangGraph全流程加入后台任务，任务已启动")

    # 9. 所有文件处理完毕，返回上传成功信息和所有TaskID
    logger.info(f"多文件上传处理完毕，共处理{len(files)}个文件，生成TaskID列表：{task_ids}")
    return {
        "code": 200,
        "message": f"Files uploaded successfully, total: {len(files)}",
        "task_ids": task_ids
    }

# --------------------------
# 核心接口：任务状态查询接口
# 前端轮询此接口获取单个任务的处理进度和状态
# 访问地址：http://localhost:8000/status/{task_id} （GET请求）
# --------------------------
@app.get("/status/{task_id}", summary="任务状态查询", description="根据TaskID查询单个文件的处理进度和全局状态")
async def get_task_progress(task_id: str):
    """
    任务状态查询接口
    前端轮询此接口（如每秒1次），获取任务的实时处理进度
    返回数据均来自内存中的任务管理字典（task_utils.py），高性能无IO

    :param task_id: 全局唯一任务ID（由/upload接口返回）
    :return: 包含任务全局状态、已完成节点、运行中节点的JSON响应
    """
    # 构造任务状态返回体
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),  # 任务全局状态：pending/processing/completed/failed
        "done_list": get_done_task_list(task_id),  # 已完成的节点/阶段列表
        "running_list": get_running_task_list(task_id)  # 正在运行的节点/阶段列表
    }
    # 记录状态查询日志，方便追踪前端轮询情况
    logger.info(
        f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}，已完成节点：{task_status_info['done_list']}")
    return task_status_info

# --------------------------
# 服务启动入口
# 直接运行此脚本即可启动FastAPI服务，无需额外执行uvicorn命令
# --------------------------
if __name__ == "__main__":
    """服务启动入口：本地开发环境直接运行"""
    logger.info("File Import Service 服务启动中...")
    # 启动uvicorn服务，绑定本地IP和8000端口，关闭自动重载（生产环境建议用workers多进程）
    uvicorn.run(
        app=app,
        host="127.0.0.1",  # 仅本地访问，生产环境改为0.0.0.0（允许所有IP访问）
        port=8000  # 服务端口
    )