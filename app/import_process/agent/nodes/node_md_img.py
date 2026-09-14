import os
import re
import base64
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque

# MinIO相关依赖
from minio.deleteobjects import DeleteObject

# 【核心改造1：移除原生OpenAI，导入LangChain工具类和多模态消息模块】
from app.clients.minio_utils import get_minio_client
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task
from langchain_core.output_parsers import StrOutputParser
# LLM客户端工具类（核心复用，替换原生OpenAI调用）
from app.lm.lm_utils import get_llm_client
# LangChain多模态依赖（消息构造+异常捕获）
from langchain.messages import HumanMessage
# 项目配置
from app.conf.minio_config import minio_config
from app.conf.lm_config import lm_config
# 项目日志工具（统一使用）
from app.core.logger import logger, node_log, step_log
# api访问限速工具
from app.utils.rate_limit_utils import apply_api_rate_limit
# 提示词加载工具
from app.core.load_prompt import load_prompt

# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

def is_supported_image(filename: str) -> bool:
    """
    判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
    :param filename: 文件名（含后缀）
    :return: 支持返回True，否则False
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

"""
    主要目标: 就是将md中的图片进行单独处理,图片转成对应的语义文本,方便后续进行切片搜索!
    主要动作: 图片 -> 图片服务器 -> (上文)图片内容(下文) -> 传递到视觉模型 -> 生成图片总结
             -> 替换md原有的图片显示 ![图片总结](图片的minio网络地址) -> state 修改md_content / md_path 新内容 -> 结束
    技术总结: minio reg正则 多模态模型 提示词
    实现步骤: 
          1. 进行任务和日志处理 
          2. 进行核心参数校验 [校验md_path/md_content/返回images的文件夹地址]
          3. 查找md中使用的图片和上下文 [传入md_content和images文件夹,返回进行模型访问准备 [(图片名,图片地址,(上文,下文))]]
          4. 进行图片内容总结和处理[调用多模态模型,总结图片内容,最终返回 图片名/总结]
          5. 上传图片到minio服务器,替换图片的本地地址和描述!返回替换后的md_content内容
          6. 备份新的md内容,改为原名称 _new.md
          7. 进行md_path和md_content内容更新(state)
          8. 返回目标结果即可
"""



@step_log("step_1_get_content")
def step_1_get_content(state) -> Tuple[str, Path, Path]:
    """
    提取和校验内容,并且返回图片的地址
    :param state:
    :return: 返回md内容,md地址,images地址
    """
    # 1.获取md地址 md_path
    md_file_path = state.get("md_path")
    if not md_file_path:
        raise ValueError("md_path参数错误,请检查输入参数!")
    md_file_obj = Path(md_file_path)
    if not md_file_obj.exists():
        raise FileNotFoundError(f"md_path参数错误,请检查输入参数! {md_file_path}")

    # 2.获取读取md_content
    if not state['md_content']:
        state['md_content'] = md_file_obj.read_text(encoding="utf-8")

    # 3.拼接图片存储地址
    images_dir_obj = md_file_obj.parent / "images"

    return state['md_content'], md_file_obj, images_dir_obj

@step_log("step_2_scan_images")
def step_2_scan_images(md_content: str, images_dir_obj: Path) -> List[Tuple[str, str, Tuple[str, str]]]:
    """
    扫描 MD 文档中的图片，匹配本地图片文件，并截取图片上下文（前后100字符）
    :param md_content: Markdown 原文内容
    :param images_dir_obj: 图片所在目录（Path 对象）
    :return: 列表 -> [(图片文件名, 图片完整路径, (上文, 下文))]
    """
    # 存储最终处理好的图片信息
    image_targets = []
    # 目录不存在（如直接导入的 MD 无图片、PDF 未抽出图片）→ 直接返回空列表，避免 WinError 3
    if not images_dir_obj.exists():
        logger.info(f"图片目录不存在，跳过图片扫描：{images_dir_obj}")
        return image_targets
    # 使用 pathlib 遍历目录（现代、安全、自带完整路径）
    for image_file in images_dir_obj.iterdir():
        img_name = image_file.name  # 图片文件名（如：test.png）
        # 过滤非图片格式
        if not is_supported_image(img_name):
            logger.warning(f"跳过非图片文件：{img_name}")
            continue
        # 正则匹配 MD 中的图片语法：![...](...图片名...)
        # re.escape 处理文件名中带特殊字符（如括号、点）导致正则爆炸的问题
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(img_name) + ".*?\)")
        items = list(pattern.finditer(md_content))
        # 没有匹配到 → 跳过
        if not items:
            logger.warning(f"图片 {img_name} 未在 MD 中引用，跳过")
            continue
        # 获取图片在 MD 中的位置
        start, end = items[0].span()
        # 截取上下文（前后各100字符）
        pre_text = md_content[max(start - 100, 0): start]
        post_text = md_content[end: min(end + 100, len(md_content))]
        context = (pre_text, post_text)
        # 组装结果：文件名、图片完整路径、上下文
        # str(image_file) = 直接获取绝对路径（Path 自带，无需拼接）
        image_targets.append((img_name, str(image_file), context))
    return image_targets

@step_log("step_3_generate_img_summaries")
def step_3_image_summary(image_targets, stem) -> Dict[str, str]:
    """
    总结图片,生成图片名.png - 对应的图片描述内容
    :param image_targets: 图片信息 [(文件名,文件地址,(上文,下文))]
    :param stem: 文件名 -> 提示词需要
    :return: 图片总结
    """
    # 1. 定义总结字典
    summaries = {}
    # 2. 定义任务队列 -> 模型访问队列 -> 限制访问次数
    # - 在 node_md_img.py 里 request_times = deque() 定义在函数内部。
    # - 所以每次调用 step_3_generate_img_summaries() 都会创建一个 新的队列 。
    # - 它只能限制“这一次函数调用里的图片循环速率”， 不是全局限流 。
    # 也就是说：
    # - 单次任务内：有效（同一批图片会被限速）
    # - 多次请求/多任务并发：彼此不共享队列，不会互相限速
    # 如果你要“全局限制”，要改成共享状态，比如：
    # - 模块级全局 deque （仅单进程有效）
    # - Redis 限流（多进程/多实例推荐，企业常用）
    # - 网关层限流（如 Nginx/API Gateway）
    requests_limiter = deque()

    for image_file,image_path,context in image_targets:
        # 访问限速问题（我们模型的限速标准 1分钟 可以访问10  限制并发访问次数..）
        # 具体要根据模型的配置 https://help.aliyun.com/zh/model-studio/rate-limit?spm=a2c4g.11186623.help-menu-2400256.d_0_0_3.29c5d355nLkkXf&scm=20140722.H_2840182._.OR_help-T_cn~zh-V_1
        apply_api_rate_limit(requests_limiter,max_requests=100)
        # 获取多模态模型对象
        vm_model = get_llm_client(model=lm_config.lv_model)
        # 准备提示词
        prompt = load_prompt(name="image_summary", root_folder=stem, image_content=context)
        # 将图片转成base64字符串
        # path.read_text()	读取文本（txt/md）	str 字符串
        # path.write_text()	写入文本	str 字符串
        # path.read_bytes()	读取二进制（图片 / 视频）	bytes 字节
        # path.write_bytes()	写入二进制（保存文件）	bytes 字节
        if isinstance(image_path, str):
            image_path = Path(image_path)
        # # 转成字符串 base64.b64encode(Path.read_bytes()) -> 转成base64字节数据格式 .decode() 转成字符串
        image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        )

        # 调用模型并获取结果
        chain = vm_model | StrOutputParser()
        summary = chain.invoke([message])
        summaries[image_file] = summary

    return summaries

@step_log("step_4_upload_images")
def step_4_upload_images_replace(image_summaries, image_targets, md_content, stem) -> str:
    """
    将图片上传到minio服务器!
    同时替代原md内容中的图片地址和描述内容!
    确保任何位置可以进行访问图片和现实
    :param image_summaries: 图片 和 总结
    :param image_targets: 图片名称 地址 和上下文
    :param md_content: 原md内容
    :param stem: md文件名
    :return: 替换后的md_content
    """
    # 1. 获取minio客户端对象
    minio_client = get_minio_client()
    # 2. 先清空原有md在minio中所在的图片
    # minio / 存储桶 / 文件名 / 图片
    object_list = minio_client.list_objects(
        bucket_name=minio_config.bucket_name,
        # 注意：{minio_config.minio_img_dir[1:]}  一定要去掉一个 /
        prefix=f"{minio_config.minio_img_dir[1:]}/{stem}",
        recursive=True
    )
    # 转化成minio的删除对象
    #
    delete_object_list = [
        DeleteObject(obj.object_name)
        for obj in object_list
    ]
    # 调用方法进行删除即可
    errors = minio_client.remove_objects(
        bucket_name=minio_config.bucket_name,
        delete_object_list=delete_object_list
    )
    for error in errors:
        logger.warning(f"删除图片失败: {error}")

    # 3.上传图片到minio服务器
    # 定义一个字段存储每张图片的信息 {图片名.minio_url地址}
    images_urls = {}
    for image_file,image_path,_ in image_targets:
        # 联网操作最好进行报错保护,避免直接单张失败直接异常
        try:
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=f"{minio_config.minio_img_dir}/{stem}/{image_file}",
                file_path=image_path,
                content_type="image/jpeg"
            )
            # 拼接完整路径 图片地址 = 协议 + 端点 + 桶名 + 对象名  http://47.94.86.115:9000/ 桶名 / 对象名
            images_urls[
                image_file] = f"http://{minio_config.endpoint}/{minio_config.bucket_name}/{minio_config.minio_img_dir}/{stem}/{image_file}"
            logger.debug(f"完成图片:{image_file}上传,URL:{images_urls[image_file]}")
        except Exception as e:
            logger.exception(f"上传图片失败：{image_file}，失败原因：{e}")
            logger.debug("继续尝试上传下一张图片!")
    #4.拼接替换的完整资料 {image_file,(summary,url)}
    images_infos = {}
    for image_file,summary in image_summaries.items():
        images_infos[image_file] = (summary,images_urls[image_file])

    #5.进行md_content内容替换
    if images_infos:
        for image_file,(summary,url) in images_infos.items():
            # 定义正则
            # ![](/xxx/xx/image_file) -> ![无所谓](无所谓image_file无所谓)
            # 正则必须加固！否则图片名带特殊符号直接炸
            # 如果图片名叫：image(1).png image[2].png
            # 里面的 () . [] 都是正则特殊符号，直接报错！
            # 加固 re.escape (图片名) 作用：把图片名里的特殊符号自动转义，正则不会炸！
            rep = re.compile(r"!\[.*?\]\(.*?"+ re.escape(image_file) +".*?\)")
            # 进行替换
            # 方法	作用	结果	你用它来干嘛
            # findall	查找所有匹配	返回匹配到的文本列表	找图片、找内容
            # finditer	查找所有匹配	返回带位置的匹配对象	截取上下文、取位置
            # sub	    查找 + 替换	返回替换后的新文本	改内容、换链接
            # md_content = rep.sub(f"![{summary}]({url})", md_content)
            # - 每匹配到一次，就执行这个函数
            # - 函数返回什么字符串，就原样拿去替换
            # - re 不再去解析里面的 \T 、 \1 这些东西
            md_content = rep.sub(lambda _: f"![{summary}]({url})", md_content)
    logger.debug(f"完成新旧md内容替换,最新内容:{md_content[:200]}")
    return md_content

@step_log("step_5_backup_md_file")
def step_5_backup_md_file(md_path_obj, new_md_content) -> str:
    """
    完成新的md的磁盘备份,并且返回新的地址!
    新的命名规则: 原名称_new.md
    :param md_path_obj:
    :param new_md_content:
    :return: 返回新地址
    """
    #   c:/xxx/xxx/xxx/xxxx/erdaye.md
    #   -》 c:/xxx/xxx/xxx/xxxx/erdaye _new.md
    # new_md_path_obj = md_path_obj.with_stem(f"{md_path_obj.stem}_new")
    new_md_path_obj = md_path_obj.parent / (md_path_obj.stem + "_new" + md_path_obj.suffix)

    new_md_path_obj.write_text(new_md_content,encoding="utf-8")

    return str(new_md_path_obj)

@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    未来要实现:
    1. 扫描 Markdown 中的图片链接。
    2. 将图片上传到 MinIO 对象存储。
    3. (可选) 调用多模态模型生成图片描述。
    4. 替换 Markdown 中的图片链接为 MinIO URL。
    """

    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    """
    # 1. 进行任务和日志处理
    add_running_task(state['task_id'],'node_md_img')
    # 2. 进行核心参数校验 [校验md_path/md_content/返回images的文件夹地址]
    md_content,md_path_obj,images_dir_obj = step_1_get_content(state)
    # 3. 查找md中使用的图片和上下文 [传入md_content和images文件夹,返回进行模型访问准备 [(图片名,图片地址,(上文,下文))]]
    image_targets = step_2_scan_images(md_content, images_dir_obj)
    # 4. 进行图片内容总结和处理[调用多模态模型,总结图片内容,最终返回 图片名/总结]
    image_summaries = step_3_image_summary(image_targets,md_path_obj.stem)
    # 5. 上传图片到minio服务器,替换图片的本地地址和描述!返回替换后的md_content内容
    new_md_content = step_4_upload_images_replace(image_summaries, image_targets , md_content, md_path_obj.stem)
    # 6. 备份新的md内容,改为原名称 _new.md
    new_md_file_path_str = step_5_backup_md_file(md_path_obj, new_md_content)
    # 7. 进行md_path和md_content内容更新(state)
    state['md_path'] = new_md_file_path_str
    state['md_content'] = new_md_content
    # 8. 返回目标结果即可
    add_done_task(state['task_id'], 'node_md_img')
    return state

if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
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
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")