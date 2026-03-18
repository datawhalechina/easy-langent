import os
import io
import contextlib
import platform
import matplotlib
# [关键] 永久设置非交互后端，防止服务器报错
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from src.data_manager import get_dataframe

load_dotenv()

# --- 解决中文乱码 ---
def configure_fonts():
    system_name = platform.system()
    if system_name == 'Windows':
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    elif system_name == 'Darwin':
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
configure_fonts()

# -----------------------------------------------------------------------------
# 工具 1: Python 计算 (回归你的单作用域逻辑，但增加 stdout 捕获)
# -----------------------------------------------------------------------------
class PythonCodeInput(BaseModel):
    py_code: str = Field(description="Python代码。可以使用变量 df。")

@tool(args_schema=PythonCodeInput)
def python_inter(py_code: str):
    """
    执行 Python 代码。
    """
    df = get_dataframe()
    if df is None:
        return "错误：当前没有加载任何数据。"

    # [回归原始逻辑]：只使用一个 env 字典，同时充当 globals 和 locals
    # 这样列表推导式就不会报错了
    env = {
        "df": df, 
        "pd": pd, 
        "np": np, 
        "result": None
    }
    
    output_buffer = io.StringIO()
    
    try:
        # 使用 contextlib 捕获 print() 的内容
        with contextlib.redirect_stdout(output_buffer):
            # [关键] 只传一个字典！
            exec(py_code, env)
            
        # 1. 优先获取 stdout (print的内容)
        stdout_content = output_buffer.getvalue().strip()
        
        # 2. 其次获取 result 变量 (如果用户写了 result=...)
        result_content = ""
        if "result" in env and env["result"] is not None:
            result_content = str(env["result"])

        # 3. 实在不行，尝试 eval 最后一行 (模仿你原始代码的逻辑)
        # 这是一个兜底策略，为了让 '2+2' 这种直接返回 4
        eval_result = ""
        if not stdout_content and not result_content:
            try:
                # 尝试计算最后一行表达式
                last_line = py_code.strip().split('\n')[-1]
                # 只有当最后一行看起来像表达式时才 eval
                if not last_line.startswith('print') and '=' not in last_line:
                    eval_result = str(eval(last_line, env))
            except:
                pass

        # 组合输出
        final_output = []
        if stdout_content: final_output.append(f"【打印输出】\n{stdout_content}")
        if result_content: final_output.append(f"【计算结果】\n{result_content}")
        if eval_result: final_output.append(f"【表达式值】\n{eval_result}")
        
        if not final_output:
            return "代码执行成功，但没有输出。请使用 print() 打印结果。"
            
        return "\n\n".join(final_output)

    except Exception as e:
        return f"代码执行报错: {e}"

# -----------------------------------------------------------------------------
# 工具 2: 绘图 (回归简单逻辑，保留路径配置)
# -----------------------------------------------------------------------------
class FigCodeInput(BaseModel):
    py_code: str = Field(description="绘图代码。需生成 fig 对象。")
    fname: str = Field(description="图像变量名，例如 'fig'。")

@tool(args_schema=FigCodeInput)
def fig_inter(py_code: str, fname: str) -> str:
    """
    执行绘图代码并保存。
    """
    df = get_dataframe()
    if df is None: return "错误：无数据。"
    
    print(f">>> 开始绘图: {fname}")
    
    # 清理画布
    plt.clf()
    plt.close('all')

    # [回归原始逻辑]：单字典作用域
    env = {
        "df": df, 
        "pd": pd, 
        "plt": plt, 
        "sns": sns
    }
    
    # 路径配置 (保留这部分，因为必须存到 static 目录前端才能看)
    # 确保这个路径和 server.py 挂载的路径一致
    save_dir = os.path.join(os.getcwd(), "static", "images")
    os.makedirs(save_dir, exist_ok=True)

    try:
        # [关键] 只传一个字典！
        exec(py_code, env)
        
        fig = env.get(fname)
        # 容错：如果用户没赋值给 fname，尝试获取当前 fig
        if not fig:
            fig = plt.gcf()
            
        if fig:
            # 文件名处理
            file_name = f"{fname}.png"
            abs_path = os.path.join(save_dir, file_name)
            
            # 保存
            fig.savefig(abs_path, bbox_inches='tight', dpi=100)
            print(f">>> 图片保存成功: {abs_path}")
            
            # 返回前端标记
            return f"IMAGE_GENERATED: {file_name}"
        else:
            return "绘图代码执行完毕，但未找到图像对象。"
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"绘图报错: {e}"
    finally:
        plt.close('all')