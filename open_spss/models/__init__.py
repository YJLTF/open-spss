"""模型注册:导入即完成全部模型注册。"""
from . import basic, dimreduc, evaluation, forecast, medical, ml, plots, quality, regression, reliability, tests  # noqa: F401

from ..core.registry import REGISTRY

CATEGORIES = REGISTRY.categories()
