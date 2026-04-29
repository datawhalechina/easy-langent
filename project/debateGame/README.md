# 智能体辩论赛主持系统

🚀 核心功能

- 多智能体协作：4个AI辩手（正方2人、反方2人）+ 1个裁判，共5个智能体参与辩论
- 多轮辩论迭代：支持2轮自由辩论，每轮参考历史发言，保持逻辑连贯
- 状态管理：使用 TypedDict 全程跟踪辩论进度、发言记录、投票结果
- 胜负判定：裁判基于论点质量、逻辑性、说服力评判胜负

🛠️ 技术栈

- 框架: LangGraph, LangChain
- 模型: DeepSeek
- 状态存储: 内存检查点 (MemorySaver)

📋 工作节点

- generate_topic: 根据用户输入话题生成一个有争议性的辩论题目
- assign_roles: 随机分配正方/反方角色给4个辩手
- opening_statement: 每个辩手生成50-100字的开篇陈词
- debate_round1: 第1轮自由辩论，各辩手相互反驳
- debate_round2: 第2轮自由辩论，继续深化论点
- closing_statement: 正反方选个辩手生成总结陈词
- judge_result: 裁判基于全程表现评判胜负

🔄 工作流程

```
辩题生成 → 角色分配 → 开篇陈词 → 自由辩论(2轮) → 总结陈词 → 裁判评判
```

📝 运行示例
📄 状态结构

```python
class DebateState(TypedDict):
    topic: str  # 辩题
    user_input: str  # 根据用户主题生成辩题
    sides: dict  # 角色分配：{agent1: ("正方"/"反方", 角色名), ...}
    opening_statements: dict  # 开篇陈词：{agent1: "陈词内容", ...}
    debate_round1: dict  # 第1轮自由辩论：{agent1: "发言", ...}
    debate_round2: dict  # 第2轮自由辩论：{agent1: "发言", ...}
    closing_statements: dict  # 总结陈词：{agent1: "陈词", ...}
    judge_result: str  # 裁判评判结果
    winner: str  # 获胜方：正方/反方/平局

```

⚠️ 注意事项

- 本项目需要有效的 DeepSeek API Key
- 辩论过程中LLM输出受模型能力影响，可能出现格式偏差
- 可通过调整 temperature 控制输出的创造性
