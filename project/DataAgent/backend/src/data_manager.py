import pandas as pd
import numpy as np
import io
import os  # 需要导入 os 来获取文件名

# 全局变量存储
GLOBAL_DF = None
GLOBAL_FILENAME = "未命名数据集" # [新增] 存储文件名

def _preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    [内部函数] 数据预处理流水线：
    1. 去除全空行列
    2. 智能类型推断 (object -> numeric)
    3. 缺失值填充
    """
    # 1. 基础清理：去除全空的行和列
    df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
    
    # 2. 智能类型转换
    # 尝试将 object 类型的列转换为数值，无法转换的变成 NaN
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # 尝试转数字，coerce 模式下无法转换的变为 NaN
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                
                # 如果转换后的非空值比例超过 50%，我们认为这列应该是数字列 (例如: "10", "20", "N/A")
                if numeric_series.notna().sum() > 0.5 * len(df):
                    df[col] = numeric_series
            except Exception:
                pass

    # 3. 缺失值处理 (简单粗暴策略，适合演示)
    # 数值列：用均值填充
    # 类别列：用 "Unknown" 填充
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna("Unknown").astype(str)

    return df

def load_csv_file(file_path: str):
    """加载并清洗 CSV 文件"""
    global GLOBAL_DF, GLOBAL_FILENAME # [修改] 引用全局文件名
    try:
        # [新增] 提取并保存文件名 (不包含完整路径，只存文件名，保护隐私且够用)
        GLOBAL_FILENAME = os.path.basename(file_path)
        
        raw_df = pd.read_csv(file_path)
        clean_df = _preprocess_data(raw_df)
        GLOBAL_DF = clean_df
        
        rows, cols = GLOBAL_DF.shape
        numeric_cols = list(GLOBAL_DF.select_dtypes(include=[np.number]).columns)
        
        return True, (f"成功加载文件【{GLOBAL_FILENAME}】！\n"
                      f"包含 {rows} 行，{cols} 列。")
    except Exception as e:
        return False, f"数据加载失败: {str(e)}"

def get_dataframe():
    """获取当前 DataFrame"""
    return GLOBAL_DF

def get_data_preview(n=10):
    """获取前 N 行数据 (处理 NaN 为 None 以便 JSON 序列化)"""
    if GLOBAL_DF is not None:
        # replace({np.nan: None}) 是为了防止前端 JSON 解析报错
        return GLOBAL_DF.head(n).replace({np.nan: None}).to_dict(orient='records')
    return []

def get_data_info():
    """获取数据摘要 (Schema)"""
    if GLOBAL_DF is not None:
        buffer = io.StringIO()
        GLOBAL_DF.info(buf=buffer)
        
        # [修改] 在返回的信息头部拼接文件名
        info_str = f"数据来源文件: {GLOBAL_FILENAME}\n" 
        info_str += "-" * 30 + "\n"
        info_str += buffer.getvalue()
        
        return info_str
    return "暂无数据"

def calculate_correlation(col1: str, col2: str):
    """
    [增强版] 计算相关性
    支持：数值 vs 数值 (Pearson), 类别 vs 数值 (Label Encoding), 类别 vs 类别
    """
    if GLOBAL_DF is None:
        return 0.0

    if col1 not in GLOBAL_DF.columns or col2 not in GLOBAL_DF.columns:
        return 0.0

    try:
        s1 = GLOBAL_DF[col1]
        s2 = GLOBAL_DF[col2]

        # 辅助函数：将序列转为数值
        def to_numeric_force(series):
            if pd.api.types.is_numeric_dtype(series):
                return series
            else:
                # 如果是字符串/类别，使用 factorize 编码 (0, 1, 2...)
                codes, uniques = pd.factorize(series)
                return pd.Series(codes)

        # 强制转换为数值序列
        v1 = to_numeric_force(s1)
        v2 = to_numeric_force(s2)

        # 计算 Pearson 相关系数
        corr = v1.corr(v2)
        
        # 处理计算结果为 NaN 的情况 (例如标准差为0)
        if pd.isna(corr):
            return 0.0
            
        return round(corr, 4)

    except Exception as e:
        print(f"相关性计算出错: {e}")
        return 0.0