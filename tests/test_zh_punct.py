"""zh_punct 标点归一化规则:中文语境全角化,数字千分位/代码/链接保留。"""
from open_spss.core.results import zh_punct


def test_comma_after_digit_converts():
    # 统计文本最常见的写法:数字后跟逗号再接内容,必须转全角
    assert zh_punct("t=2.1,p=0.03") == "t=2.1，p=0.03"
    assert zh_punct("均值 10863.4237,组「男」均值 10623.878") == "均值 10863.4237，组「男」均值 10623.878"


def test_thousands_separator_kept():
    assert zh_punct("共 1,000 人,计 1,234,567 元") == "共 1,000 人，计 1,234,567 元"


def test_code_span_kept():
    assert zh_punct("置信区间 `[-1.5, 2.0]` 如下:") == "置信区间 `[-1.5, 2.0]` 如下："


def test_markdown_link_kept():
    link = "[方法说明](http://a.com/help?q=1)"
    assert zh_punct(f"详见 {link} 谢谢") == f"详见 {link} 谢谢"


def test_misc_full_width():
    assert zh_punct("如下:") == "如下："
    assert zh_punct("正在启动...") == "正在启动……"
    assert zh_punct("(方差齐),显著!") == "（方差齐），显著！"
    assert zh_punct("已经全角，不再变化") == "已经全角，不再变化"
