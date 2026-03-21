import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langserve import add_routes
from dotenv import load_dotenv

# 1. 强制加载环境变量 (确保 API Key 能被读取)
load_dotenv(override=True)

# 导入项目模块
# 注意：确保你的目录结构正确，且在根目录下运行 python -m src.server
from src.agent import graph
from src.data_manager import (
    load_csv_file, 
    get_dataframe, 
    get_data_preview, 
    calculate_correlation
)

# =============================================================================
# 2. 初始化 FastAPI 应用
# =============================================================================
app = FastAPI(
    title="Data Agent Backend",
    version="1.0",
    description="DeepSeek Data Analysis Agent API with LangGraph"
)

# =============================================================================
# 3. 配置 CORS (解决 Figma 跨域的关键)
# =============================================================================
app.add_middleware(
    CORSMiddleware,
    # ❌ 不要用 allow_origins=["*"]，因为我们需要 allow_credentials=True
    allow_origins=[], 
    
    # ✅ 使用正则动态匹配：
    # 1. Figma 动态沙盒域名 (https://...figma.site)
    # 2. 本地调试 (localhost, 127.0.0.1)
    allow_origin_regex=r"https://.*\.figma\.site|http://localhost.*|http://127\.0\.0\.1.*",
    
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# 4. 挂载静态文件 (用于图片展示)
# =============================================================================
# 确保目录存在
static_dir = os.path.join(os.getcwd(), "static")
images_dir = os.path.join(static_dir, "images")
os.makedirs(images_dir, exist_ok=True)

# 挂载 /static 路径
# 前端通过 http://localhost:8002/static/images/xxx.png 访问图片
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# =============================================================================
# 5. 定义请求模型
# =============================================================================
class CorrelationRequest(BaseModel):
    col1: str
    col2: str

# =============================================================================
# 6. 核心接口定义
# =============================================================================

@app.get("/")
async def root():
    return {"status": "ok", "message": "Data Agent Backend is running"}

# --- 接口 A: 上传 CSV ---
@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """
    前端上传 CSV 文件，后端保存并加载到内存 DataFrame
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="只支持 CSV 文件")

    # 临时保存路径
    temp_dir = "temp_data"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)

    try:
        # 写入磁盘
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 调用 data_manager 加载数据
        success, message = load_csv_file(file_path)
        
        if success:
            # 获取预览数据返回给前端
            preview = get_data_preview()
            return JSONResponse(content={
                "status": "success",
                "message": message,
                "preview": preview,  # 前端表格数据源
                "filename": file.filename
            })
        else:
            raise HTTPException(status_code=500, detail=message)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 接口 B: 计算相关性 ---
@app.post("/calculate-correlation")
async def get_correlation(request: CorrelationRequest):
    """
    前端点击两个变量时调用此接口，快速返回相关系数
    """
    # 调用 data_manager 中的纯逻辑函数
    result = calculate_correlation(request.col1, request.col2)
    
    # 生成简单的描述文案
    desc = "无相关性"
    try:
        corr_val = float(result)
        if abs(corr_val) > 0.8: desc = "极强相关"
        elif abs(corr_val) > 0.6: desc = "强相关"
        elif abs(corr_val) > 0.4: desc = "中等相关"
        elif abs(corr_val) > 0.2: desc = "弱相关"
    except:
        pass

    return {
        "status": "success",
        "correlation": result,
        "description": desc
    }

# --- 接口 C: Agent 对话 (LangServe) ---
# 自动挂载 /agent/invoke, /agent/stream 等接口
add_routes(
    app,
    graph,
    path="/agent",
)

# =============================================================================
# 7. 启动入口
# =============================================================================
if __name__ == "__main__":
    print(">>> 启动 Data Agent 后端服务...")
    print(">>> API 文档地址: http://localhost:8002/docs")
    print(">>> 图片存储路径:", images_dir)
    
    # 启动服务，端口 8002
    uvicorn.run("src.server:app", host="0.0.0.0", port=8002, reload=True)