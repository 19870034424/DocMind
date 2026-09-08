"""Agentic RAG — ReAct 循环主控

核心流程:
1. 初始化消息（system prompt + user question）
2. 循环（max 8 次）:
   a. 调用 LLM (带 tools 定义)
   b. 有 tool_calls → 执行工具 → yield step 事件 → 追加到消息
   c. 无 tool_calls → 检测 clarify/answer → break
3. 流式输出最终答案
"""
import json
import logging
import re
from collections.abc import AsyncGenerator

from openai import OpenAI

from config import settings
from models.agent import AgentStep, AgentStepType, Source
from services.agent_prompts import AGENT_SYSTEM_PROMPT, TOOLS
from services.agent_tools import execute_tool, format_tool_result

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 8
CLARIFY_PATTERNS = [
    "请问", "你想了解", "你能告诉", "请告诉", "你想看",
    "你指的是", "你说的是", "能否明确", "具体是哪",
    "你希望", "你更关心", "需要你确认", "请选择",
    "你想总结", "你想深入", "我可以帮你",
]


def _detect_clarify_intent(text: str) -> bool:
    """检测 LLM 输出是否包含追问意图"""
    if not text:
        return False
    text = text.strip()
    if len(text) < 8:
        return False
    if "?" in text or "？" in text:
        return any(p in text for p in CLARIFY_PATTERNS)
    return False


def _parse_tool_calls(choice) -> list[dict]:
    """安全解析 LLM 返回的 tool_calls"""
    if choice is None or not hasattr(choice, "tool_calls") or not choice.tool_calls:
        return []
    parsed = []
    for tc in choice.tool_calls:
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tool arguments: {tc.function.arguments[:100]}")
            continue
        parsed.append({
            "id": tc.id,
            "name": tc.function.name,
            "arguments": args,
        })
    return parsed


def _build_llm_client() -> tuple[OpenAI, str]:
    """构建 LLM 客户端，复用 config 中的配置"""
    if settings.llm_provider == "qwen":
        client = OpenAI(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
        )
        model = settings.qwen_model
    else:
        client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
        model = settings.deepseek_model
    return client, model


class AgentLoop:
    """ReAct Agent 循环"""

    def __init__(
        self,
        doc_id: str,
        question: str,
        max_iterations: int = MAX_ITERATIONS,
    ):
        self.doc_id = doc_id
        self.question = question
        self.max_iterations = max_iterations
        self.iteration_count = 0
        self.steps: list[AgentStep] = []
        self.used_sources: list[Source] = []

        # 初始化消息列表
        self.messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"文档 ID: {doc_id}\n\n用户问题: {question}"},
        ]

    def _should_force_answer(self) -> bool:
        return self.iteration_count >= self.max_iterations

    async def run(self) -> AsyncGenerator[dict, None]:
        """运行 Agent 循环，逐步骤 yield SSE 事件 dict

        返回格式: {"event": "step"|"chunk"|"sources"|"done", "data": {...}}
        """
        step_index = 0

        # Phase 1: ReAct 循环
        while not self._should_force_answer():
            self.iteration_count += 1
            step_index += 1

            # 推送规划步骤
            yield {
                "event": "step",
                "data": AgentStep(
                    step_index=step_index,
                    type=AgentStepType.PLAN,
                    content=f"分析问题并决定下一步操作...（第 {self.iteration_count}/{self.max_iterations} 轮）",
                    detail={},
                ).model_dump(mode="json"),
            }

            choice = await self._call_llm()
            if choice is None:
                yield {
                    "event": "step",
                    "data": AgentStep(
                        step_index=step_index,
                        type=AgentStepType.ERROR,
                        content="LLM 调用失败，请重试",
                        detail={},
                    ).model_dump(mode="json"),
                }
                return

            tool_calls = _parse_tool_calls(choice)

            if tool_calls:
                for tc in tool_calls:
                    step_index += 1
                    tool_name = tc["name"]
                    tool_args = tc.get("arguments", {})

                    # ask_user 特殊处理
                    if tool_name == "ask_user":
                        yield {
                            "event": "step",
                            "data": AgentStep(
                                step_index=step_index,
                                type=AgentStepType.CLARIFY,
                                content=tool_args.get("question", "请问您需要什么帮助？"),
                                detail={
                                    "tool_name": "ask_user",
                                    "options": tool_args.get("options", []),
                                },
                            ).model_dump(mode="json"),
                        }
                        self.messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": "ask_user",
                                    "arguments": json.dumps(tool_args, ensure_ascii=False),
                                },
                            }],
                        })
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": "[等待用户回复...]",
                        })
                        return

                    # 推送 tool_call 事件
                    yield {
                        "event": "step",
                        "data": AgentStep(
                            step_index=step_index,
                            type=AgentStepType.TOOL_CALL,
                            content=f"调用工具: {tool_name}",
                            detail={"tool_name": tool_name, "arguments": tool_args},
                        ).model_dump(mode="json"),
                    }

                    # 执行工具
                    result = execute_tool(self.doc_id, tool_name, tool_args)
                    formatted = format_tool_result(tool_name, result)

                    # 记录引用来源
                    if tool_name in ("search_document", "search_keyword") and "hits" in result:
                        for hit in result["hits"]:
                            excerpt = hit.get("text", "")[:200]
                            if excerpt and not any(s.excerpt == excerpt for s in self.used_sources):
                                self.used_sources.append(Source(
                                    heading=hit.get("heading", ""),
                                    page=hit.get("page", 0),
                                    excerpt=excerpt,
                                    score=hit.get("score", 0.0),
                                ))

                    # 推送 tool_result 事件
                    yield {
                        "event": "step",
                        "data": AgentStep(
                            step_index=step_index,
                            type=AgentStepType.TOOL_RESULT,
                            content=formatted,
                            detail={
                                "tool_name": tool_name,
                                "result_summary": formatted[:200],
                                "hits_count": result.get("total", 0) if "total" in result else None,
                            },
                        ).model_dump(mode="json"),
                    }

                    # 追加工具结果到消息
                    self.messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_args, ensure_ascii=False),
                            },
                        }],
                    })
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": formatted,
                    })
            else:
                content = choice.content or ""
                if _detect_clarify_intent(content):
                    yield {
                        "event": "step",
                        "data": AgentStep(
                            step_index=step_index,
                            type=AgentStepType.CLARIFY,
                            content=content,
                            detail={},
                        ).model_dump(mode="json"),
                    }
                    self.messages.append({"role": "assistant", "content": content})
                    return
                break

        if self._should_force_answer():
            yield {
                "event": "step",
                "data": AgentStep(
                    step_index=step_index + 1,
                    type=AgentStepType.PLAN,
                    content=f"已达到最大检索轮数 ({self.max_iterations})，基于已有信息生成答案",
                    detail={},
                ).model_dump(mode="json"),
            }

        # Phase 3: 生成最终答案
        async for event in self._generate_answer():
            yield event

    async def _call_llm(self):
        """调用 LLM 获取下一步决策"""
        client, model = _build_llm_client()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=self.messages,
                tools=TOOLS,
                temperature=0.0,
                max_tokens=4096,
            )
            return resp.choices[0].message if resp.choices else None
        except Exception as e:
            logger.error(f"LLM call failed in AgentLoop: {e}")
            return None

    async def _generate_answer(self) -> AsyncGenerator[dict, None]:
        """流式生成最终答案"""
        step_index = len(self.steps) + 1

        # 先自检
        cite_texts = [s.excerpt for s in self.used_sources[:10]]
        reflect_result = execute_tool(
            self.doc_id, "reflect_on_answer",
            {"draft_answer": "[待生成]", "cited_sources": cite_texts},
        )
        reflect_formatted = format_tool_result("reflect_on_answer", reflect_result)
        yield {
            "event": "step",
            "data": AgentStep(
                step_index=step_index,
                type=AgentStepType.REFLECT,
                content=reflect_formatted,
                detail=reflect_result,
            ).model_dump(mode="json"),
        }

        # 流式输出答案
        answer_prompt = (
            "请基于以上的检索结果和工具调用历史，用中文简洁准确地回答用户问题。"
            "如检索结果不足以回答，请明确说明。"
        )
        if not reflect_result.get("passed", False):
            answer_prompt += "\n\n注意：自检发现问题，请特别注意引用文档来源，不要编造。"

        self.messages.append({"role": "user", "content": answer_prompt})

        client, model = _build_llm_client()
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=self.messages,
                stream=True,
                temperature=0.0,
                max_tokens=2048,
            )
            full_answer = ""
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    text = delta.content
                    # 过滤 think 标签
                    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
                    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
                    if text:
                        full_answer += text
                        yield {"event": "chunk", "data": {"text": text}}

            sources_data = [s.model_dump() for s in self.used_sources[:10]]
            yield {"event": "sources", "data": sources_data}

            total_steps = len(self.steps) + step_index
            yield {
                "event": "done",
                "data": {
                    "total_steps": total_steps,
                    "iterations": self.iteration_count,
                    "follow_up_suggestions": self._generate_follow_ups(full_answer),
                },
            }
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            yield {"event": "done", "data": {"error": str(e)}}

    def _generate_follow_ups(self, answer: str) -> list[str]:
        """根据答案生成追问建议"""
        suggestions = []
        if any(kw in self.question for kw in ("区别", "对比", "vs", "比较")):
            suggestions.append("可以详细解释其中一个吗？")
        if len(self.used_sources) >= 3:
            suggestions.append("还有哪些相关概念值得了解？")
        if self.iteration_count >= self.max_iterations - 2:
            suggestions.append("检索可能不够全面，需要换关键词重新搜索吗？")
        return suggestions

    def resume(self, user_response: str):
        """用户回复追问后恢复循环"""
        self.messages.append({
            "role": "user",
            "content": f"[用户回复] {user_response}",
        })
