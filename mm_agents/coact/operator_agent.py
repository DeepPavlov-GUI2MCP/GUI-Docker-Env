# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import base64
import json
import os
import traceback
from urllib.parse import urlparse
from typing import Any, Callable, Dict, Literal, Optional, Union

from openai import OpenAI
from desktop_env.desktop_env import DesktopEnv

from .autogen.llm_config import LLMConfig
from .autogen.agentchat.conversable_agent import ConversableAgent
from .autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent

from .cua_agent import (
    DEFAULT_CUA_MODEL,
    DEFAULT_WEBPAGE_BACKEND,
    READ_WEBPAGE_TOOL,
    _build_openai_client,
    read_webpage,
    run_cua,
    validate_cua_model,
)
from .coding_agent import TerminalProxyAgent, CODER_SYSTEM_MESSAGE


class OrchestratorAgent(MultimodalConversableAgent):
    """(In preview) Captain agent, designed to solve a task with an agent or a group of agents."""

    CALL_GUI_AGENT_TOOL = {
        "type": "function",
        "function": {
            "name": "call_gui_agent",
            "description": """Let a OS Operator to solve a task. OS operator can operate the computer by clicking and typing (not accurate in dense UI). Require detailed task description.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "[REQUIRED] A detailed task to be solved with step-by-step guidance.",
                    },
                },
            },
        },
    }

    CALL_CODING_AGENT_TOOL = {
        "type": "function",
        "function": {
            "name": "call_coding_agent",
            "description": """(You MUST use this first) Let a programmer to solve a task. Coding agent can write python and bash code with many tools to solve a task. Require detailed task and environment description.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "[REQUIRED] A detailed task to be solved.",
                    },
                    "environment": {
                        "type": "string",
                        "description": "[REQUIRED] The environment description of the coding agent. It should be a detailed description of the system state, including the opened files, the running processes, etc.",
                    }
                },
            },
        },
    }

    CALL_WEB_SEARCH_TOOL = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": """Search the public web for task-relevant instructions, documentation, and troubleshooting guidance. Use this before deciding a task is infeasible when research is required.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "[REQUIRED] The search query with task and app context.",
                    },
                    "allowed_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of domains to restrict results to, such as official docs or Stack Exchange sites.",
                    },
                },
                "required": ["query"],
            },
        },
    }
    CALL_READ_WEBPAGE_TOOL = {
        "type": "function",
        "function": {
            "name": READ_WEBPAGE_TOOL["name"],
            "description": READ_WEBPAGE_TOOL["description"],
            "parameters": READ_WEBPAGE_TOOL["parameters"],
        },
    }

    CALL_API_SUMMARY_AGENT_TOOL = {
        "type": "function",
        "function": {
            "name": "call_api_summary_agent",
            "description": """Let a API summary agent to summarize the API response. API summary agent can summarize the API response. Require detailed API response.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "[REQUIRED] A url of the API response."},
                },
            },
        },
    }

    DEFAULT_DESCRIPTION = ""

    # This is used to prompt the LLM to summarize the conversation history between CaptainAgent's tool execution history
    DEFAULT_SUMMARY_PROMPT = "Read the following conversation history between an expert and a group of agent experts, summarize the conversation history. Your summarization should include the initial task, the experts' plan and the attempt, finally the results of the conversation. If the experts arrived at a conclusion, state it as it is without any modification."

    def __init__(
        self,
        name: str,
        system_message: Optional[str] = None,
        llm_config: Optional[Union[LLMConfig, dict[str, Any], Literal[False]]] = None,
        is_termination_msg: Optional[Callable[[dict[str, Any]], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        code_execution_config: Optional[Union[dict[str, Any], Literal[False]]] = False,
        description: Optional[str] = DEFAULT_DESCRIPTION,
        enable_coding_agent: bool = False,
        enable_web_search_tool: bool = False,
        enable_read_webpage_tool: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            name,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            description=description,
            **kwargs,
        )

        if system_message is None:
            self.update_system_message("")
        else:
            self.update_system_message(system_message)

        self.enable_coding_agent = enable_coding_agent
        if self.enable_coding_agent:
            self.update_tool_signature(self.CALL_CODING_AGENT_TOOL, is_remove=False)
        self.update_tool_signature(self.CALL_GUI_AGENT_TOOL, is_remove=False)
        if enable_web_search_tool:
            self.update_tool_signature(self.CALL_WEB_SEARCH_TOOL, is_remove=False)
        if enable_read_webpage_tool:
            self.update_tool_signature(self.CALL_READ_WEBPAGE_TOOL, is_remove=False)
        # self.assistant.update_tool_signature(self.CALL_API_SUMMARY_AGENT_TOOL, is_remove=False)  # TODO: add this tool later


class OrchestratorUserProxyAgent(MultimodalConversableAgent):
    """(In preview) A proxy agent for the captain agent, that can execute code and provide feedback to the other agents."""

    DEFAULT_AUTO_REPLY = "Thank you! Note that the user's task is: {user_instruction}. Please continue the task. If you think the everything is solved, please reply me only with 'TERMINATE'. But once you think the task is impossible to solve, please reply me only with 'INFEASIBLE'."

    DEFAULT_USER_PROXY_AGENT_DESCRIPTIONS = {
        "ALWAYS": "An attentive HUMAN user who can answer questions about the task, and can perform tasks such as running Python code or inputting command line commands at a Linux terminal and reporting back the execution results.",
        "TERMINATE": "A user that can run Python code or input command line commands at a Linux terminal and report back the execution results.",
        "NEVER": "A computer terminal that can running Python scripts (provided to it quoted in ```python code blocks), or sh shell scripts (provided to it quoted in ```sh code blocks), or the conversation history and result of a group of agents",
    }

    CONVERSATION_REVIEW_PROMPT = """You are looking for a conversation history between a user and an agent.
    Given the conversation history below, summarize the conversation history in a concise way.

    - Conversation history:
    {chat_history}

    - Response template (markdown format):
    # Summarize of the conversation history
    ...(include the middle terminal output. They are important.)

    # Final result
    ...
    """

    def __init__(
        self,
        name: str,
        is_termination_msg: Optional[Callable[[dict[str, Any]], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        code_execution_config: Optional[Union[dict[str, Any], Literal[False]]] = {},
        default_auto_reply: Optional[Union[str, dict[str, Any]]] = DEFAULT_AUTO_REPLY,
        llm_config: Optional[Union[LLMConfig, dict[str, Any], Literal[False]]] = False,
        system_message: Optional[Union[str, list]] = "",
        description: Optional[str] = None,

        # GUI Agent config
        provider_name: str = "docker",
        path_to_vm: str = None,
        observation_type: str = "screenshot",
        screen_width: int = 1920,
        screen_height: int = 1080,
        sleep_after_execution: float = 1.0,
        truncate_history_inputs: int = 51,
        cua_max_steps: int = 50,
        coding_max_steps: int = 30,
        history_save_dir: str = "",
        llm_model: str = "o4-mini",
        gui_model: str = DEFAULT_CUA_MODEL,
        orchestrator_model: str = "o3",
        region: str = "us-east-1",
        client_password: str = "",
        user_instruction: str = "",
        enable_web_search_tool: bool = False,
        enable_read_webpage_tool: bool = False,
        enable_coding_agent: bool = False,
        prompt_mode: str = "default",
        task_source: Optional[str] = None,
        orchestrator_client_kwargs: Optional[Dict[str, Any]] = None,
        orchestrator_backend_config: Optional[Dict[str, Any]] = None,
        gui_client_kwargs: Optional[Dict[str, Any]] = None,
        coding_llm_config: Optional[LLMConfig] = None,
    ):
        description = (
            description if description is not None else self.DEFAULT_USER_PROXY_AGENT_DESCRIPTIONS[human_input_mode]
        )
        super().__init__(
            name=name,
            system_message=system_message,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply.format(user_instruction=user_instruction),
            description=description,
        )
        function_map = {
            "call_gui_agent": lambda **args: self._call_gui_agent(**args, screen_width=screen_width, screen_height=screen_height),
        }
        if enable_read_webpage_tool:
            function_map["read_webpage"] = lambda **args: self._read_webpage(**args)
        if enable_web_search_tool:
            function_map["web_search"] = lambda **args: self._web_search(**args)
        if enable_coding_agent:
            function_map["call_coding_agent"] = lambda **args: self._call_coding_agent(**args)
        self.register_function(function_map=function_map)
        self._code_execution_config = code_execution_config
        self.cua_config = {
            "max_steps": cua_max_steps,
            "sleep_after_execution": sleep_after_execution,
            "truncate_history_inputs": truncate_history_inputs,
        }
        self.region = region
        self.client_password = client_password

        screen_size = (screen_width, screen_height)
        snapshot_name = "init_state"
        if provider_name == "aws":
            from desktop_env.providers.aws.manager import IMAGE_ID_MAP

            snapshot_name = IMAGE_ID_MAP[region].get(screen_size, IMAGE_ID_MAP[region][(1920, 1080)])

        self.env = DesktopEnv(
            path_to_vm=path_to_vm,
            action_space="pyautogui",
            provider_name=provider_name,
            os_type="Ubuntu",
            region=region,
            snapshot_name=snapshot_name,
            screen_size=screen_size,
            headless=True,
            require_a11y_tree=observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"],
            enable_proxy=True,
            client_password=client_password
        )

        self.history_save_dir = history_save_dir
        self.cua_call_count = 0
        self.coding_call_count = 0
        self.coding_max_steps = coding_max_steps
        self.llm_config = llm_config
        self.llm_model = llm_model
        self.orchestrator_model = orchestrator_model
        validate_cua_model(gui_model)
        self.gui_model = gui_model
        self.enable_web_search_tool = enable_web_search_tool
        self.enable_read_webpage_tool = enable_read_webpage_tool
        self.enable_coding_agent = enable_coding_agent
        self.prompt_mode = prompt_mode
        self.task_source = task_source
        self.orchestrator_client_kwargs = orchestrator_client_kwargs or {}
        self.orchestrator_backend_config = orchestrator_backend_config or {}
        self.gui_client_kwargs = gui_client_kwargs or {}
        self.coding_llm_config = coding_llm_config or LLMConfig(api_type="openai", model=self.llm_model)
        self.web_search_client: OpenAI = _build_openai_client(
            api_key=self.orchestrator_backend_config.get("api_key"),
            base_url=self.orchestrator_backend_config.get("base_url"),
        )
        self.web_search_model = self.orchestrator_model

    def reset(self, task_config: dict[str, Any]):
        obs = self.env.reset(task_config=task_config)
        print(f"VM started on localhost:{self.env.vnc_port}", flush=True)
        return obs

    def _call_gui_agent(self, task: str, screen_width: int = 1920, screen_height: int = 1080) -> str:
        """Run a GUI agent to solve the task."""
        cua_path = os.path.join(self.history_save_dir, f'cua_output_{self.cua_call_count}')
        if not os.path.exists(cua_path):
            os.makedirs(cua_path)
        try:
            history_inputs, result, cost, raw_transcript, tool_events = run_cua(
                self.env,
                task,
                save_path=cua_path,
                max_steps=self.cua_config["max_steps"],
                cua_model=self.gui_model,
                base_url=self.gui_client_kwargs.get("base_url"),
                api_key=self.gui_client_kwargs.get("api_key"),
                screen_width=screen_width,
                screen_height=screen_height,
                sleep_after_execution=self.cua_config["sleep_after_execution"],
                truncate_history_inputs=self.cua_config["truncate_history_inputs"],
                client_password=self.client_password,
                prompt_mode=self.prompt_mode,
                task_source=self.task_source,
            )
            screenshot = self.env.controller.get_screenshot()

            with open(os.path.join(cua_path, "history_inputs.json"), "w") as f:
                json.dump(history_inputs, f)
            with open(os.path.join(cua_path, "responses.json"), "w") as f:
                json.dump(raw_transcript, f)
            with open(os.path.join(cua_path, "tool_events.json"), "w") as f:
                json.dump(tool_events, f)
            with open(os.path.join(cua_path, "result.txt"), "w") as f:
                f.write(result)
            with open(os.path.join(cua_path, "cost.txt"), "w") as f:
                f.write(str(cost))
            self.cua_call_count += 1

        except Exception as e:
            return f"# Response from GUI agent error ({type(e).__name__}): {e}\n{traceback.format_exc()}"

        if "TERMINATE" in result:
            result = result.replace("TERMINATE", "").strip()
            if result == "":
                result = "Task completed. Please check the screenshot."
        elif "IDK" in result:
            result = result.replace("IDK", "").strip()
        else:
            result = f"I didn't complete the task and I have to go. Now I'm working on \"{result}\", please check the current screenshot."
        return f"# Response from GUI agent: {result}<img data:image/png;base64,{base64.b64encode(screenshot).decode('utf-8')}>"

    def _web_search(self, query: str, allowed_domains: Optional[list[str]] = None) -> str:
        cleaned_query = str(query).strip()
        if not cleaned_query:
            return "# Web search error: missing query."

        normalized_domains = [
            domain.strip().lower()
            for domain in (allowed_domains or [])
            if isinstance(domain, str) and domain.strip()
        ]
        if self.prompt_mode == "inspect-source-first" and self.task_source:
            source_domain = urlparse(self.task_source).netloc.strip().lower()
            if source_domain and source_domain not in normalized_domains:
                normalized_domains.insert(0, source_domain)

        tool: dict[str, Any] = {"type": "web_search", "search_context_size": "medium"}
        if normalized_domains:
            tool["filters"] = {"allowed_domains": normalized_domains}

        try:
            response = self.web_search_client.responses.create(
                model=self.web_search_model,
                tools=[tool],
                input=cleaned_query,
                include=["web_search_call.action.sources"],
            )
        except Exception as exc:
            return f"# Web search error: {type(exc).__name__}: {exc}"

        response_dump = response.model_dump(mode="json") if hasattr(response, "model_dump") else {}
        response_text = getattr(response, "output_text", "").strip()
        sources: list[tuple[str, str]] = []

        for item in response_dump.get("output", []):
            if item.get("type") == "web_search_call":
                for source in item.get("action", {}).get("sources", []) or []:
                    title = str(source.get("title") or source.get("url") or "").strip()
                    url = str(source.get("url") or "").strip()
                    if url and (title, url) not in sources:
                        sources.append((title, url))
            if item.get("type") == "message":
                for content_item in item.get("content", []) or []:
                    if content_item.get("type") != "output_text":
                        continue
                    for annotation in content_item.get("annotations", []) or []:
                        if annotation.get("type") != "url_citation":
                            continue
                        title = str(annotation.get("title") or annotation.get("url") or "").strip()
                        url = str(annotation.get("url") or "").strip()
                        if url and (title, url) not in sources:
                            sources.append((title, url))

        lines = ["# Web search result", response_text or "No text summary returned."]
        if normalized_domains:
            lines.extend(["", f"Restricted domains: {', '.join(normalized_domains)}"])
        if sources:
            lines.append("")
            lines.append("Sources:")
            lines.extend(f"- {title}: {url}" if title else f"- {url}" for title, url in sources[:8])
        return "\n".join(lines)

    def _read_webpage(
        self,
        url: str,
        max_chars: int = 12000,
        backend: str = DEFAULT_WEBPAGE_BACKEND,
    ) -> str:
        result = read_webpage(url=url, max_chars=max_chars, backend=backend)
        lines = ["# Webpage inspection"]
        if result.get("title"):
            lines.append(f"Title: {result['title']}")
        if result.get("url"):
            lines.append(f"URL: {result['url']}")
        lines.append(f"Backend: {result.get('backend', backend)}")
        if result.get("error"):
            lines.extend(["", f"Error: {result['error']}"])
            return "\n".join(lines)
        lines.extend(["", result.get("content", "") or "No readable content returned."])
        if result.get("truncated"):
            lines.append("")
            lines.append(f"Truncated to {max_chars} characters.")
        return "\n".join(lines)
    
    def _call_coding_agent(self, task: str, environment: str) -> str:
        """Run a coding agent to solve the task."""
        if not self.enable_coding_agent:
            return "# Coding agent is disabled for this run."
        default_auto_reply = "I'm a code interpreter and I can only execute your code or end the conversation. If you think the problem is solved, please reply me only with 'TERMINATE'."
        try:
            screenshot = self.env.controller.get_screenshot()
            coding_agent = MultimodalConversableAgent(
                name="coding_agent",
                llm_config=self.coding_llm_config,
                system_message=CODER_SYSTEM_MESSAGE.format(CLIENT_PASSWORD=self.client_password),
            )
            code_interpreter = TerminalProxyAgent(
                name="code_interpreter",
                human_input_mode="NEVER",
                code_execution_config={
                    "use_docker": False,
                    "timeout": 300,
                    "last_n_messages": 1,
                },
                max_consecutive_auto_reply = None,
                default_auto_reply = default_auto_reply,
                description = None,
                is_termination_msg=lambda x: x.get("content", "") and x.get("content", "")[0]["text"].lower() == "terminate",
                env=self.env,
            )
            code_interpreter.initiate_chat(
                recipient=coding_agent,
                message=f"# Task\n{task}\n\n# Environment\n{environment}<img data:image/png;base64,{base64.b64encode(screenshot).decode('utf-8')}>",
                max_turns=self.coding_max_steps,
            )
        
            chat_history = []
            key = list(code_interpreter.chat_messages.keys())[0]
            chat_messages = code_interpreter.chat_messages[key]
            for item in chat_messages:
                for content in item['content']:
                    if content['type'] == 'image_url':
                        content['image_url']['url'] = '<image>'
                chat_history.append(item)
            
            if not os.path.exists(os.path.join(self.history_save_dir, f'coding_output_{self.coding_call_count}')):
                os.makedirs(os.path.join(self.history_save_dir, f'coding_output_{self.coding_call_count}'))
                
            with open(os.path.join(self.history_save_dir, f'coding_output_{self.coding_call_count}', "chat_history.json"), "w") as f:
                json.dump(chat_history, f)
            self.coding_call_count += 1

            # Review the group chat history
            summarizer = ConversableAgent(
                name="summarizer",
                llm_config=self.coding_llm_config,
                system_message=self.CONVERSATION_REVIEW_PROMPT,
            )
            summarized_history = summarizer.generate_oai_reply(
                messages=[
                    {
                        "role": "user",
                        "content": self.CONVERSATION_REVIEW_PROMPT.format(chat_history=chat_history),
                    }
                ]
            )[1]
        except Exception:
            return f"# Call coding agent error: {traceback.format_exc()}"

        screenshot = self.env.controller.get_screenshot()
        return f"# Response from coding agent: {summarized_history}"
