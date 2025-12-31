from langchain.output_parsers import (
    StructuredOutputParser,
    ResponseSchema,
    RetryOutputParser
)
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://api.deepseek.com"

# 1. 模型
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3
)

# 2. 结构化解析器
schemas = [
    ResponseSchema(name="tool_name", description="工具名称"),
    ResponseSchema(name="difficulty", description="学习难度：简单 / 中等 / 复杂"),
]
base_parser = StructuredOutputParser.from_response_schemas(schemas)

# 3. Retry 解析器（关键）
parser = RetryOutputParser.from_llm(
    llm=llm,
    parser=base_parser,
    max_retries=2
)

# 4. Prompt（只负责“格式约束”）
prompt = PromptTemplate(
    template="""
请介绍 1 个 LangChain 开发工具。
{format_instructions}
""",
    input_variables=[],
    partial_variables={
        "format_instructions": base_parser.get_format_instructions()
    }
)

# 5. 调用
prompt_value = prompt.format_prompt()
response = llm.invoke(prompt_value)


# 6. 解析（失败会自动 Retry）
result = parser.parse_with_prompt(
    response.content,
    prompt_value
)

print(result)
