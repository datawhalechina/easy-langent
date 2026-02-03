# 小说创作智能体

🚀 核心功能
- 循环迭代逻辑：支持在“基础设定”和“大纲章节”阶段进行人工审核，若不满意可输入修改意见重新生成。
- 状态管理：使用 TypedDict 和 MemorySaver 全程跟踪创作进度和数据。
- 逐章生成：突破长文本生成限制，根据大纲逻辑逐章创作，保证情节连贯性。

🛠️ 技术栈
- 框架: LangGraph, LangChain
- 模型: DeepSeek 
- 状态存储: 内存检查点 (MemorySaver)

📋 工作节点
- get_user_input: 接收题材、风格等初始需求。    
- generate_basic_setting: 生成标题、角色设定和背景。
- confirm_basic_setting (中断点): 人工审核设定，支持反馈修改。
- generate_outline_chapter: 构建故事大纲和章节结构（至少8章）。
- confirm_outline_chapter (中断点): 人工审核大纲逻辑，支持反馈修改。
- generate_complete_novel: 自动迭代生成各章节正文并汇总。

