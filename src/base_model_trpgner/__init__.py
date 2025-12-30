"""
base-model-trpgner - HydroRoll TRPG NER 模型 SDK

这是一个用于 TRPG（桌上角色扮演游戏）日志命名实体识别的 Python SDK。

基本用法:
    >>> from base_model_trpgner import TRPGParser
    >>> parser = TRPGParser()
    >>> result = parser.parse("风雨 2024-06-08 21:44:59 剧烈的疼痛...")
    >>> print(result)
    {'metadata': {'speaker': '风雨', 'timestamp': '2024-06-08 21:44:59'}, 'content': [...]}

训练功能（需要额外安装）:
    >>> pip install base-model-trpgner[train]
    >>> from base_model_trpgner.training import train_ner_model
    >>> train_ner_model(conll_data="./data", output_dir="./model")
"""

from base_model_trpgner.inference import TRPGParser, parse_line, parse_lines

try:
    from importlib.metadata import version
    __version__ = version("base_model_trpgner")
except Exception:
    __version__ = "0.1.3"

__all__ = [
    "__version__",
    "TRPGParser",
    "parse_line",
    "parse_lines",
]


def get_version():
    return __version__
