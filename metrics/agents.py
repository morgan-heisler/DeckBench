import logging
import sys
import asyncio
import httpx
from contextlib import AsyncExitStack

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    cast,
)

## AWorld agent
from aworld.agents.llm_agent import Agent
# from aworld.runner import Runners
from aworld.core.task import Task
from aworld.config.conf import AgentConfig, ModelConfig, TaskConfig
from aworld.core.agent.swarm import Swarm, TeamSwarm, GraphBuildType
from aworld.core.context.prompts import BasePromptTemplate
from aworld.runners.callback.decorator import reg_callback
# from aworld.logs.util import logger
from aworld.config.conf import ClientType

## OpenAI
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled, gen_trace_id, trace
from agents.mcp import MCPServer, MCPServerStdio
import httpx
from contextlib import AsyncExitStack
from agents import enable_verbose_stdout_logging


from .prompts import coherence_system_prompt, content_system_prompt, tsbench_system_prompt, coherence_prompt, content_prompt, tsbench_prompt


# -------------------------
# MCP config
# -------------------------
filesystem_mcp_config = {
    "mcpServers": {
        "filesystem": {
            "command": "mcp-server-filesystem",
            "args": [
                "--project-dir",
                "/root",
            ]
        },

    }
}

#### Initialize Agents

def init_agent_openai(name='Assistant', system_prompt="You are an helpful assistant.",  mcp_config={}, model_name=None, model_server=None, api_key='dummy', timeout = 100, tool_timeout = 300): 
    from agents import Agent
    from agents.model_settings import ModelSettings
    from agents.models.default_models import (
        get_default_model_settings,
        gpt_5_reasoning_settings_required,
        is_gpt_5_default,
    )

    # Configure the 'openai.agents' logger
    logger = logging.getLogger("openai.agents")
    logger.setLevel(logging.INFO)

    # Add a handler to output the logs (e.g., to standard error)
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # You can do the same for tracing logs if needed
    logging.getLogger("openai.agents.tracing").setLevel(logging.INFO) 
    logging.getLogger("openai.agents.tracing").addHandler(handler)

    # enable_verbose_stdout_logging() # enable logging

    def prepare_mcp_servers(mcp_dict: Dict, timeout_seconds:float):
        mcp_servers = []
        if "mcpServers" in mcp_dict.keys():
            mcpServers = mcp_dict["mcpServers"]

            for server_name in mcpServers.keys():
                params = mcpServers[server_name]

                if 'type' in params.keys() and params['key'] == 'sse':
                    mcp_server = MCPServerSse(
                        name= server_name,
                        params = prams,
                        client_session_timeout_seconds = timeout_seconds
                    )
                else:
                    mcp_server = MCPServerStdio(
                        name= server_name,
                        params = params,
                        client_session_timeout_seconds = timeout_seconds
                    )
                mcp_servers.append(mcp_server)
        return mcp_servers

    if model_server == 'None' or model_server == None:
        model_server = ''

    if model_name == 'None' or model_name == '' or model_name == None:
        return None
    
    model_settings = ModelSettings()

    custom_async_http_client = httpx.AsyncClient(verify=False)
    client = AsyncOpenAI(base_url=model_server, api_key=api_key, http_client=custom_async_http_client, timeout= timeout)
    set_tracing_disabled(disabled=True)

    mcp_servers = prepare_mcp_servers(mcp_config, timeout_seconds=tool_timeout)
    agent = Agent(
        name=name,
        instructions= system_prompt,
        model=OpenAIChatCompletionsModel(model=model_name, openai_client=client),
        model_settings=model_settings, #
        mcp_servers= mcp_servers,
    )

    return agent

# @staticmethod
def init_agent_aworld(name='Agent', system_prompt="You are an helpful assistant.", agent_prompt=None, description="", mcp_config={}, model_name=None, model_server=None, api_key='dummy'): 

    # =========================
    # AWorld Multi-Agent Framework
    # =========================
    from aworld.agents.llm_agent import Agent
    from aworld.config.conf import AgentConfig, ModelConfig


    def get_gpt_model_major_version(model_name):
        major_version = 0
        if model_name.startswith('gpt') and len(model_name.split('-')) > 1:
            version_str = model_name.split('-')[1]
            major_version_str = version_str[0]

            try:
                major_version = int(major_version_str)
            except ValueError:
                pass
            except TypeError:
                pass

        return major_version

    if model_server == 'None' or model_server == None:
        model_server = ''

    if model_name == 'None' or model_name == '' or model_name == None:
        return None

    llm_temperature = 0.1 
    gpt_major_version = 0
    if model_name.startswith('gpt'):
        gpt_major_version = get_gpt_model_major_version(model_name)

    if gpt_major_version >=5: # only for gpt-5
        llm_temperature = 1

    # print('====== model name : ', model_name, ' , ulr : ', model_server, ' , api_key : ', api_key)

    # Create a shared model configuration
    model_config = ModelConfig(
        llm_provider="openai",
        llm_model_name= model_name,
        llm_base_url= model_server,
        llm_api_key=api_key, 
        llm_temperature=llm_temperature,

        ext_config= {'ssl_verify': False, #False for Huawei MaaS
                    },
    )
    # Use the shared model config in agent configuration
    agent_config = AgentConfig(
        llm_config= model_config,
        use_vision=False,
    )

    agent = Agent(
        name= name, 
        conf= agent_config,
        desc=description,
        system_prompt= system_prompt,
        agent_prompt=agent_prompt,
        mcp_servers= mcp_config.get("mcpServers", {}).keys(),
        mcp_config= mcp_config,    
    )

    return agent




#### Inference
async def inference_openai(agent, user_prompt):

    agent_answer = ''

    try: 
        mcp_servers = agent.mcp_servers
        async with AsyncExitStack() as stack:
            for mcp_server in mcp_servers:
                await stack.enter_async_context(mcp_server)

            result = await Runner.run(starting_agent=agent, input=user_prompt)
            agent_answer = result.final_output

        # print('')
        # print('')
        # print('')
        # print('===== OpenAIAgent answer : ')
        # print('')
        # print(agent_answer)


        if '</think>' in agent_answer:
            agent_answer = agent_answer.split('</think>')[-1].strip()
        else:
            agent_answer = agent_answer

        # if '</thinking>' in agent_answer:
        #     agent_answer = agent_answer.split('</thinking>')[-1].strip()
        # else:
        #     agent_answer = agent_answer

        if "json" in agent_answer:
            agent_answer = agent_answer.split('```json')[-1]
            agent_answer = agent_answer.split('```')[0]        

        if "```html" in agent_answer:
            agent_answer = agent_answer.split('```html')[-1]
            agent_answer = agent_answer.split('```')[0]

        if agent_answer.startswith('html'):
            agent_answer = agent_answer.removeprefix('html')
    except Exception as e:
        print(f"Error inference with OpenAIAgent: {e}")

    return agent_answer

# @staticmethod
def inference_aworld(agent, user_prompt):

    from aworld.runner import Runners

    # Create agent group with collaborative workflow
    group = Swarm(topology=[(agent)])

    # task_response = Runners.sync_run(
    task_response =  asyncio.run( 
        Runners.run(
            input= user_prompt,  
            swarm=group,
        )
    )
    agent_answer = task_response.answer

    if '</think>' in agent_answer:
        agent_answer = agent_answer.split('</think>')[-1].strip()
    else:
        agent_answer = agent_answer

    if "json" in agent_answer:
        agent_answer = agent_answer.split('```json')[-1]
        agent_answer = agent_answer.split('```')[0]        

    if "```html" in agent_answer:
        agent_answer = agent_answer.split('```html')[-1]
        agent_answer = agent_answer.split('```')[0]

    if agent_answer.startswith('html'):
        agent_answer = agent_answer.removeprefix('html')

    return agent_answer

# @staticmethod
def initialize_llm_judge_agents(model_name=None, model_server=None, api_key='dummy', agent_type='OpenAIAgent'):
    ## INITIALIZE AGENTS - we only want to do this once
    OPENAIAGENT_TIMEOUT =  500

    if agent_type == 'OpenAIAgent':
        coherence_agent = init_agent_openai(name='Assistant', system_prompt=coherence_system_prompt,  mcp_config=filesystem_mcp_config, model_name=model_name, model_server=model_server, api_key=api_key, timeout = OPENAIAGENT_TIMEOUT, tool_timeout = 500)
    elif agent_type == 'AWorld':
        coherence_agent = init_agent_aworld(name = "AWorld Agent", system_prompt=coherence_system_prompt, mcp_config=filesystem_mcp_config, model_name=model_name, model_server=model_server, api_key=api_key)
    # coherence_group = Swarm(topology=[(coherence_agent)])

    if agent_type == 'OpenAIAgent':
        content_agent = init_agent_openai(name='Assistant', system_prompt=content_system_prompt,  mcp_config=filesystem_mcp_config, model_name=model_name, model_server=model_server, api_key=api_key, timeout = OPENAIAGENT_TIMEOUT, tool_timeout = 500)
    elif agent_type == 'AWorld':
        content_agent = init_agent_aworld(name = "AWorld Agent", system_prompt=content_system_prompt, mcp_config=filesystem_mcp_config, model_name=model_name, model_server=model_server, api_key=api_key)
    # content_group = Swarm(topology=[(content_agent)])

    if agent_type == 'OpenAIAgent':
        tsbench_agent = init_agent_openai(name='Assistant', system_prompt=tsbench_system_prompt,  mcp_config=filesystem_mcp_config, model_name=model_name, model_server=model_server, api_key=api_key, timeout = OPENAIAGENT_TIMEOUT, tool_timeout = 500)
    elif agent_type == 'AWorld':
        tsbench_agent = init_agent_aworld(name = "AWorld Agent", system_prompt=tsbench_system_prompt, mcp_config=filesystem_mcp_config, model_name=model_name, model_server=model_server, api_key=api_key)
    # tsbench_group = Swarm(topology=[(tsbench_agent)])

    return coherence_agent, content_agent, tsbench_agent

