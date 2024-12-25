from typing import Literal
def format_float_or_none(float_num: str, width: int, sig: int, align: Literal['<', '>', "^"]) -> str:
    if float_num is None:
        return f"{'None':{align}{width}}"
    else:
        return f"{float_num:{align}{width}.{sig}f}"

def format_str_or_none(string_inp: str, width: int, align: Literal['<', '>', "^"]) -> str:
    if string_inp is None:
        return f"{'None':{align}{width}}"
    else:
        return f"{string_inp:{align}{width}}"

def format_list_float_or_none(list_float: list, width: int, sig: int, align: Literal['<', '>', "^"]) -> str:
    if list_float is None:
        return f"{'None':{align}{width}}"
    else:
        return f"[{', '.join([f'{x:{align}{width}.{sig}f}' if x > 1e-4 else f'{x:{align}{width}.{sig}e}' for x in list_float])}]"
