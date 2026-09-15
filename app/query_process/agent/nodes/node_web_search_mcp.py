import asyncio
import os
import json
import sys
from agents.mcp import MCPServerSse # pip install openai-agents
from agents.mcp import MCPServerStreamableHttp # pip install openai-agents

from app.conf.bailian_mcp_config import mcp_config
from app.utils.task_utils import add_running_task,add_done_task

DASHSCOPE_BASE_URL_SSE = 'https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp'
DASHSCOPE_API_KEY = 'sk-ws-H.EYIHHDR.Y0z5.MEQCIG9pli3ycfRb8bJwIG-1aDBUmt0nvdhbIQ057eGWxkydAiBfixQnEeBHoQQy6i3inQxW5VHu11j91SJlVlzaLi7L6g'

async def mcp_call(query):
    # 初始化 MCP
    search_mcp = MCPServerSse(
        name="search_mcp",
        params={
            "url": DASHSCOPE_BASE_URL_SSE,
            "headers": {"Authorization": DASHSCOPE_API_KEY},
            "timeout": 300,
            "sse_read_timeout": 300
        }
    )

    try:
        await search_mcp.connect()
        # 直接调用工具
        result = await search_mcp.call_tool(
            tool_name="bailian_web_search",
            arguments={"query": query, "count": 5}
            # arguments={"query": "今天北京的天气情况", "count": 5}
        )
        return result
    finally:
        await search_mcp.cleanup()

async def mcp_call_streamable(query):
    search_mcp = MCPServerStreamableHttp(
        name="search_mcp",
        params={
            "url": "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp",
            "headers": {"Authorization": DASHSCOPE_API_KEY},
            "timeout": 300,
            "sse_read_timeout": 300,
            "terminate_on_close": True,
        },
        max_retry_attempts=2,
    )
    try:
        await search_mcp.connect()
        result = await search_mcp.call_tool(
            tool_name="bailian_web_search",
            arguments={"query": query, "count": 5},
        )
        return result
    finally:
        await search_mcp.cleanup()



def node_web_search_mcp(state):
    """
    节点功能，调用外部搜索引擎补充信息
    :param state:
    :return:
    """
    print("---node_web_search_mcp处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    query = state.get("rewritten_query", "")
    docs = []
    # 如果没有查询内容，直接返回
    if query:
        result = asyncio.run(mcp_call_streamable(query))
        if result:
            pages = json.loads(result.content[0].text).get("pages") or []
            # 统一输出结构化结果，供后续 rerank/引用使用
            # 每条：{title, url, snippet}

            for item in pages:
                snippet = (item.get("snippet") or "").strip()
                url = (item.get("url") or "").strip()
                title = (item.get("title") or "").strip()
                if not snippet:
                    continue
                docs.append({"title": title, "url": url, "snippet": snippet})

            print("MCP 搜索结果:", docs)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    if docs:
        return {"web_search_docs": docs}
    return {}

from dotenv import load_dotenv

if __name__ == '__main__':
    load_dotenv()
    test_state = {
        "session_id": "abc",
        "rewritten_query": "小王子里玫瑰代表了什么？",
        "is_streamable": True
    }

    # 调用 websearch_node 函数
    result_state = node_web_search_mcp(test_state)

    # 验证结果
    print("测试结果:")
    print(f"查询内容: {test_state.get('rewritten_query')}")

    # 输出搜索结果
    search_results = result_state.get('web_search_docs', [])
    print(f"搜索结果数量: {len(search_results)}")
    print("search_results", search_results)


