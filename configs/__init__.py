import sys
from pathlib import Path

def load_config(config_path=None):
    """动态加载配置文件"""
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            # 动态导入自定义配置
            sys.path.insert(0, str(config_file.parent))
            config = __import__(config_file.stem).DatasetConfig
            sys.path.pop(0)
            return config
    # 使用默认配置
    from .dataset_config import DatasetConfig
    return DatasetConfig