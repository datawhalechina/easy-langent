import os
from typing import TypedDict, NotRequired, Optional, List, Dict

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph, StateGraph, END

load_dotenv()
API_KEY = os.getenv("ZHIPU_API_KEY")
BASE_URL = os.getenv("ZHIPU_BASE_URL")
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="glm-5.2",
    temperature=0.7
)

# 定义小说创作状态
class NovelCreationState(TypedDict):
    # 用户初始输入
    user_requirement: str # 用户需求

    # 第二阶段：小说题目与主要角色、情节摘要
    novel_title: NotRequired[Optional[str]] #小说题目
    main_characters: NotRequired[Optional[List[Dict[str, str]]]] # 主要角色
    plot_overview: NotRequired[Optional[str]] # 情节摘要
    setting_modify_requirement: Optional[str] # 基础设定修改需求

    # 用户审核确认
    is_setting_confirmed: NotRequired[Optional[bool]] # 确认设定
    setting_confirmed_count: NotRequired[int] # 基础设定重试次数
    is_outline_confirmed: NotRequired[Optional[bool]] # 确认大纲
    outline_confirmed_count: NotRequired[int] # 基础设定重试次数

    # 第三阶段：大纲与章节
    novel_outline: NotRequired[Optional[str]] # 小说大纲
    chapter_structure: NotRequired[Optional[List[Dict[str, str]]]] # 章节结构（章节名，章节概括）
    outline_modify_requirement: Optional[str] # 大纲与章节修改需求

    # 第四阶段：小说生成
    complete_novel: NotRequired[Optional[str]] # 最终小说生成

    # 进度追踪字段
    current_stage: NotRequired[str] # 当前阶段
    chapter_generated_count: NotRequired[int] # 已生成章节数

# 打印信息函数
def print_process_progress(step, message):
    """统一进度打印工具"""
    print(f"\n{'='*20} {step} {'='*20}")
    print(f"⏳ {message}")
    print('-' * 50)

def print_chapter_progress(current, total):
    """打印章节生成进度（索引从0开始，显示时自动+1）"""
    print(f"\n📖 正在生成第 {current + 1}/{total} 章...")
    print("-" * 50)

# 节点1:用户输入节点
def user_input_node(state: NovelCreationState):
    print(">>>请输入你的小说创作需求（示例：科幻类型，主角是计算机专业大学生，要有AI相关的反转情节，篇幅简短）：")
    user_input = input(">>>输入：")
    return {
        "user_requirement": user_input
    }

# 节点2:LLM初始生成节点
def generate_basic_setting(state: NovelCreationState):
    print_process_progress("设定生成","(开始生成题目/角色/情节)")
    agent = ChatPromptTemplate.from_template("""
        请根据用户需求生成小说基础设定，要求：
        1. 小说题目：1-2个备选，简洁有吸引力
        2. 主要角色：至少3个，格式为「姓名：性格描述」
        3. 情节概述：100-200字，清晰说明故事整体走向
        
        用户需求：{user_requirement}
        用户修改需求：{setting_modify_requirement}（若为"无"则忽略此项）

        输出格式（严格遵循）：
        题目：xxx
        主要角色：
        - 姓名1：性格描述1
        - 姓名2：性格描述2
        - 姓名3：性格描述3
        情节概述：xxx
    """) | llm
    result = agent.invoke({
        "user_requirement": state["user_requirement"],
        "setting_modify_requirement": state.get("setting_modify_requirement", "无"),
    })
    setting_content = result.content.strip()

    # 解析结果
    lines = setting_content.split("\n")
    novel_title = None
    main_characters = []
    plot_overview = None
    for line in lines:
        if line.startswith("题目："):
            novel_title = line.replace("题目：","").strip()
        elif line.startswith("主要角色："):
            continue
        elif line.startswith("- "):
            name, desc = line.replace("- ","").split("：",1)
            main_characters.append({
                '姓名': name,
                '性格描述': desc
            })
        elif line.startswith("情节概述："):
            plot_overview = line.replace("情节概述：","").strip()

    # 展示设定
    print("\n===== 生成的小说基础设定 =====")
    print(f"题目：{novel_title}")
    print("主要角色：")
    for char in main_characters:
        print(f"- {char['姓名']} : {char['性格描述']}")
    print(f"情节概述：{plot_overview}")
    print(f"\n===== 重试次数：{state.get('setting_confirmed_count', 0)} =====\n")

    current_stage = "设定生成"
    print_process_progress("设定生成","(完成)")

    return {
        "novel_title": novel_title,
        "main_characters": main_characters,
        "plot_overview": plot_overview,
        "current_stage": current_stage
    }

# 节点3:用户确认小说基础设定
def confirm_basic_setting(state: NovelCreationState):
    print("\n===== 人工审核 - 基础设定确认环节 =====")

    is_setting_confirmed = False
    while True:
        confirm = input(">>>是否确认以上基础设定？（YES/NO）")
        if confirm.lower() == "yes":
            is_setting_confirmed = True
            print(">>>基础设定已确认，进入下一个阶段！")
            return {
                "is_setting_confirmed": is_setting_confirmed
            }
        elif confirm.lower() == "no":
            print(">>>请输入你的修改需求(如：修改角色名/调整情节/更换题目)：")
            modify_content = input(">>>请输入：")
            print(">>>正在根据你的需求修改基础设定...")

            agent = ChatPromptTemplate.from_template(
                """
                请根据用户的原始需求和修改需求，更新小说基础设定：
                原始需求：{user_requirement}
                修改需求：{setting_modify_requirement}
                输出格式（严格遵循）：
                题目：xxx
                主要角色：
                - 姓名1：性格描述1
                - 姓名2：性格描述2
                - 姓名3：性格描述3
                情节概述：xxx
                """
            ) | llm
            result = agent.invoke({
                "user_requirement": state["user_requirement"],
                "setting_modify_requirement": modify_content
            })
            setting_content = result.content.strip()

            # 重新解析
            lines = setting_content.split("\n")
            novel_title = None
            main_characters = []
            plot_overview = None

            for line in lines:
                if line.startswith("题目："):
                    novel_title = line.replace("题目：", "").strip()
                elif line.startswith("主要角色："):
                    continue
                elif line.startswith("- "):
                    name, desc = line.replace("- ", "").split("：", 1)
                    main_characters.append({"姓名": name, "性格描述": desc})
                elif line.startswith("情节概述："):
                    plot_overview = line.replace("情节概述：", "").strip()

                # 再次展示并确认
            print("\n===== 修改后的基础设定 =====")
            print(f"题目：{novel_title}")
            print("主要角色：")
            for char in main_characters:
                print(f"- {char['姓名']}：{char['性格描述']}")
            print(f"情节概述：{plot_overview}")

            # 重试次数判断逻辑
            setting_confirmed_count = state.get("setting_confirmed_count", 0) + 1
            print(f"\n===== 重试次数：{setting_confirmed_count} =====\n")
            if setting_confirmed_count >= MAX_REBASIC_COUNT:
                return {
                    "novel_title": novel_title,
                    "main_characters": main_characters,
                    "plot_overview": plot_overview,
                    "is_setting_confirmed": is_setting_confirmed,
                    "setting_confirmed_count": setting_confirmed_count
                }

            while True:
                reconfirm = input(">>>是否确认修改后的设定？(YES/NO)：")
                if reconfirm.lower() == "yes":
                    is_setting_confirmed = True
                    print(">>> 基础设定已确认！")
                    return {
                        "novel_title": novel_title,
                        "main_characters": main_characters,
                        "plot_overview": plot_overview,
                        "is_setting_confirmed": is_setting_confirmed,
                        "setting_confirmed_count": setting_confirmed_count
                    }
                elif reconfirm.lower() == "no": # 再次不通过，通过路由进行重新生成
                    print(">>>请输入你的修改需求(如：修改角色名/调整情节/更换题目)：")
                    modify_content = input(">>>请输入：")
                    print(">>>正在根据你的需求修改基础设定...")
                    setting_confirmed_count += 1
                    return {
                        "is_setting_confirmed": is_setting_confirmed,
                        "setting_confirmed_count": setting_confirmed_count,
                        "setting_modify_requirement": modify_content
                    }
                else:
                    print(">>>输入有误，请重新输入！！！")

        else:
            print(">>>输入有误，请重新输入！！！")

# 节点4:生成小说大纲与章节结构
def generate_outline_chapter(state: NovelCreationState):
    if not state.get("is_setting_confirmed",False):
        raise ValueError(">>> 基础设定未确认，无法生成大纲！")
    print_process_progress("大纲生成","(开始生成大纲/章节结构)")

    agent = ChatPromptTemplate.from_template(
        """
        请根据已确认的小说基础设定，生成：
        1. 小说整体大纲：200-300字，清晰说明故事的开端、发展、高潮、结局
        2. 章节结构：至少8章，格式为「章节X：章节情节概述（1-2句话）」，章节间逻辑连贯
        
        基础设定：
        题目：{novel_title}
        主要角色：{main_characters}
        情节概述：{plot_overview}
        用户修改请求：{outline_modify_requirement}（若为"无"则忽略此项）
        
        输出格式（严格遵循）：
        整体大纲：xxx
        章节结构：
        - 章节1：xxx
        - 章节2：xxx
        ...
        """
    ) | llm

    # 格式化角色信息
    char_str = "\n".join([f"{c['姓名']}：{c['性格描述']}" for c in state["main_characters"]])
    result = agent.invoke({
        "novel_title": state["novel_title"],
        "main_characters": char_str,
        "plot_overview": state["plot_overview"],
        "outline_modify_requirement": state.get("outline_modify_requirement","无"),
    })
    outline_content = result.content.strip()

    # 解析结果
    lines = outline_content.split("\n")
    novel_outline = None
    chapter_structure = []

    for line in lines:
        if line.startswith("整体大纲："):
            novel_outline = line.replace("整体大纲：","").strip()
        elif line.startswith("章节结构："):
            continue
        elif line.startswith("- "):
            chapter_name,chapter_desc = line.replace("- ","").split("：",1)
            chapter_structure.append({
                "章节名": chapter_name,
                "章节概括": chapter_desc
            })

    print("\n===== 生成的小说大纲与章节结构 =====")
    print(f"整体大纲：{novel_outline}")
    print("章节结构：")
    for chapter in chapter_structure:
        print(f"- {chapter['章节名']}：{chapter['章节概括']}")
    print(f"\n===== 重试次数：{state.get('outline_confirmed_count', 0)} =====\n")

    current_stage = "大纲生成"

    print_process_progress("大纲生成","(完成)")
    return {
        "novel_outline": novel_outline,
        "chapter_structure": chapter_structure,
        "current_stage": current_stage
    }

# 节点5:用户确认小说章节设定
def confirm_outline_chapter(state: NovelCreationState):
    print("\n===== 人工审核 - 大纲与章节结构确认环节 =====")

    is_outline_confirmed = False
    while True:
        print(">>>是否确认以上大纲与章节结构？(YES/NO)：")
        confirm = input(">>>请输入：")
        if confirm.lower() == "yes":
            is_outline_confirmed = True
            print(">>>大纲与章节结构已确认，进入小说生成阶段！")
            return {
                "is_outline_confirmed": is_outline_confirmed
            }
        elif confirm.lower() == "no":
            print(">>>请输入你的修改需求（如：调整章节顺序/修改某章概括/增减章节数）：")
            modify_content = input(">>>请输入：")
            print(">>>正在根据你的需求修改大纲与章节结构...")

            char_str = "\n".join([f"{c['姓名']}：{c['性格描述']}" for c in state["main_characters"]])
            agent = ChatPromptTemplate.from_template(
                """
                请根据已确认的基础设定和用户修改需求，更新小说大纲与章节结构：
                基础设定：
                题目：{novel_title}
                主要角色：{main_characters}
                情节概述：{plot_overview}
                修改需求：{modify_content}
            
                输出格式（严格遵循）：
                整体大纲：xxx
                章节结构：
                - 章节1：xxx
                - 章节2：xxx
            ...
                """
            ) | llm
            result = agent.invoke({
                "novel_title": state["novel_title"],
                "main_characters": char_str,
                "plot_overview": state["plot_overview"],
                "modify_content": modify_content
            })
            outline_content = result.content.strip()

            # 重新解析
            lines = outline_content.split("\n")
            novel_outline = None
            chapter_structure = []
            for line in lines:
                if line.startswith("整体大纲："):
                    novel_outline = line.replace("整体大纲：", "").strip()
                elif line.startswith("章节结构："):
                    continue
                elif line.startswith("- 章节"):
                    chapter_name, chapter_desc = line.replace("- ", "").split("：", 1)
                    chapter_structure.append({"章节名": chapter_name, "章节概括": chapter_desc})

            # 再次展示并确认
            print("\n===== 修改后的大纲与章节结构 =====")
            print(f"整体大纲：{novel_outline}")
            print("章节结构：")
            for chapter in chapter_structure:
                print(f"- {chapter['章节名']}：{chapter['章节概括']}")

            # 重试次数判断逻辑
            outline_confirmed_count = state.get("outline_confirmed_count", 0) + 1
            print(f"\n===== 重试次数：{outline_confirmed_count} =====\n")
            if outline_confirmed_count >= MAX_REBASIC_COUNT:
                return {
                    "novel_outline": novel_outline,
                    "chapter_structure": chapter_structure,
                    "is_outline_confirmed": is_outline_confirmed,
                    "outline_confirmed_count": outline_confirmed_count
                }

            while True:
                reconfirm = input(">>>是否确认修改后的大纲与章节结构？（YES/NO）：")
                if reconfirm.lower() == "yes":
                    is_outline_confirmed = True
                    print(">>>大纲与章节结构已确认！")
                    return {
                        "novel_outline": novel_outline,
                        "chapter_structure": chapter_structure,
                        "is_outline_confirmed": is_outline_confirmed,
                        "outline_confirmed_count": outline_confirmed_count
                    }
                elif reconfirm.lower() == "no":
                    print(">>>请输入你的修改需求（如：调整章节顺序/修改某章概括/增减章节数）：")
                    modify_content = input(">>>请输入：")
                    print(">>>正在根据你的需求修改大纲与章节结构...")
                    outline_confirmed_count += 1
                    return {
                        "is_outline_confirmed": is_outline_confirmed,
                        "outline_confirmed_count": outline_confirmed_count,
                        "outline_modify_requirement": modify_content
                    }
                else:
                    print(">>>输入有误，请重新输入！！！")
        else:
            print(">>>输入有误，请重新输入！！！")

# 节点6:按章节生成小说
def generate_complete_novel(state: NovelCreationState):
    if not state.get("is_outline_confirmed",False):
        raise ValueError(">>>大纲与章节未确认，无法生成小说！")
    print_process_progress("小说生成","(开始逐章生成正文)")

    # 初始化进度
    chapter_generated_count = 0
    chapter_total = len(state["chapter_structure"])
    print_chapter_progress(0,chapter_total)

    # 格式化基础信息
    char_str = "\n".join([f"{c['姓名']}：{c['性格描述']}" for c in state["main_characters"]])
    novel_title = state["novel_title"]
    novel_basic_info = f"""
    小说题目：{novel_title}
    主要角色：{char_str}
    整体大纲：{state["novel_outline"]}
    """

    # 完整内容
    full_novel_content = f"# {novel_title}\n\n## 小说核心设定\n{novel_basic_info.replace('    ', '')}\n\n---\n"

    # 单章生成prompt
    chapter_prompt = PromptTemplate(
        template="""
        请根据小说的核心设定、整体大纲，生成指定章节的正文内容，要求：
        1. 内容严格遵循该章节的情节概述，细节丰富，符合小说创作风格
        2. 角色性格与基础设定一致，对话自然，动作、心理描写贴合角色
        3. 章节开头标注章节名，结尾做轻微过渡，为下一章铺垫
        4. 单章字数控制在200-400字，语言流畅，情节连贯
        5. 每一章节的情节内容要连贯起来，比如：第二章情节内容是第一章情节发展过来的，第三章情节是由第二章情节发展过来的。
        
        小说核心设定：{novel_basic_info}
        当前生成章节：{chapter_name}
        本章节情节概述：{chapter_desc}
        已生成章节数：{generated_chapter_num}/{total_chapter}
        
        输出格式：直接输出生成的章节正文，无需额外说明
        """,
        input_variables=["novel_basic_info","chapter_name","chapter_desc","generated_chapter_num","total_chapter"]
    )

    # 逐章生成
    for idx, chapter in enumerate(state["chapter_structure"],1):
        chapter_name = chapter['章节名']
        chapter_desc = chapter['章节概括']
        print(f"\n【生成中】{chapter_name}...")

        # 调用LLM生成单章内容
        chapter_result = llm.invoke(chapter_prompt.format(
            novel_basic_info = novel_basic_info,
            chapter_name = chapter_name,
            chapter_desc = chapter_desc,
            generated_chapter_num = idx,
            total_chapter = chapter_total
        ))
        chapter_content = chapter_result.content.strip()

        # 拼接内容
        full_novel_content += f"\n{chapter_content}\n\n---\n"
        # 更新进度
        chapter_generated_count = idx
        print_chapter_progress(chapter_generated_count,chapter_total)
        print(f"【生成完成】{chapter_name}：\n{chapter_content}\n" + "-"*50)

    # 补充结尾
    full_novel_content += f"\n### 小说完本（总章节数：{chapter_total} | 创作基于用户需求：{state['user_requirement']}）"
    complete_novel = full_novel_content
    current_stage = "小说生成"

    # 最终进度展示
    print_process_progress("小说生成","(完成)")
    print(f"\n逐章内容生成完成！小说共{chapter_total}章，总字数>=2000字")
    return {
        "complete_novel": complete_novel,
        "current_stage": current_stage,
        "chapter_generated_count": chapter_generated_count,
    }

MAX_REBASIC_COUNT = 3
MAX_REOUTLINE_COUNT = 3

# 构建图
def build_novel_creation_graph() -> CompiledStateGraph:
    # 初始化状态图
    graph = StateGraph(NovelCreationState)
    # 添加节点
    graph.add_node("user",user_input_node)
    graph.add_node("generate_basic_setting",generate_basic_setting)
    graph.add_node("confirm_basic_setting",confirm_basic_setting)
    graph.add_node("generate_outline_chapter",generate_outline_chapter)
    graph.add_node("confirm_outline_chapter",confirm_outline_chapter)
    graph.add_node("generate_complete_novel",generate_complete_novel)

    # 定义跳转逻辑
    graph.set_entry_point("user")
    graph.add_edge("user","generate_basic_setting")
    graph.add_edge("generate_basic_setting","confirm_basic_setting")

    # 设置用户审核基础设定跳转逻辑
    def setting_confirm_router(state: NovelCreationState):
        if state.get("is_setting_confirmed",False):
            return "generate_outline_chapter"
        else:
            if state.get("setting_confirmed_count",0) >= MAX_REBASIC_COUNT: # 路由判断是否达到重试次数
                return END
            else :
                return "generate_basic_setting"
    graph.add_conditional_edges(
        source="confirm_basic_setting",
        path=setting_confirm_router
    )

    graph.add_edge("generate_outline_chapter","confirm_outline_chapter")

    # 设置用户审核大纲与章节结构跳转逻辑
    def outline_confirm_router(state: NovelCreationState):
        if state.get("is_outline_confirmed",False):
            return "generate_complete_novel"
        else:
            if state.get("outline_confirmed_count",0) >= MAX_REOUTLINE_COUNT: # 路由判断是否达到重试次数
                return END
            else :
                return "generate_outline_chapter"
    graph.add_conditional_edges(
        source="confirm_outline_chapter",
        path=outline_confirm_router
    )

    graph.add_edge("generate_complete_novel",END)

    checkpointer = MemorySaver()
    compiled_graph = graph.compile(
        checkpointer=checkpointer,
        # 个人建议：小说创作需要人机交互，建议不要设置中断点，除非你希望在某些阶段手动干预
        # interrupt_before=["confirm_basic_setting","confirm_outline_chapter"]
    )
    return compiled_graph

if __name__ == "__main__":
    novel_creation_graph = build_novel_creation_graph()
    config = {"configurable": {"thread_id": "novel_creation_1"}}
    init_state: NovelCreationState = {
        "user_requirement" : "",
        "setting_modify_requirement" : "",
        "setting_confirmed_count" : 0,
        "outline_modify_requirement" : "",
        "outline_confirmed_count" : 0,
    }
    result = novel_creation_graph.invoke(init_state, config=config)
    novel_content = result.get('complete_novel')
    
    if novel_content:
        print("\n===== 小说出版啦！！！=====\n")
        print(novel_content)
        novel_title = result.get('novel_title') or "未命名小说"
        filename = f"{novel_title}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(novel_content)
        print(f"\n📁 完整小说已保存到: {filename}")
    else:
        print("\n===== 流程已终止（达到最大重试次数）=====\n")
        print(f"当前阶段：{result.get('current_stage', '未知')}")
        print(f"基础设定重试次数：{result.get('setting_confirmed_count', 0)}")
        print(f"大纲重试次数：{result.get('outline_confirmed_count', 0)}")