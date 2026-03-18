# save_openapi.py (放在根目录运行)
import json
from src.server import app

# 导出 OpenAPI 规范
with open("openapi.json", "w", encoding="utf-8") as f:
    json.dump(app.openapi(), f, indent=2, ensure_ascii=False)

print("✅ openapi.json 已生成！")