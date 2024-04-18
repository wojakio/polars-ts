from typing import Dict


def _posix_map() -> Dict[str, str]:
    replacements = {
        r"#": "__0HASH0__",
        r"!": "__0BANG0__",
        r",": "__0COMMA0__",
        r" ": "__0SPACE0__",
        r"\t": "__0TAB0__",
        r"\n": "__0NEWLINE0__",
        r"*": "__0ASTERIX0__",
        r"£": "__0POUND0__",
        r"$": "__0DOLLAR0__",
        r"\"": "__0DQUOTE0__",
        r"'": "__0SQUOTE0__",
        r"`": "__0BTICK0__",
        r"|": "__0PIPE0__",
        r"=": "__0EQ0__",
        r"+": "__0PLUS0__",
        r"^": "__0HAT0__",
        r"(": "__0LPAREN0__",
        r")": "__0RPAREN0__",
        r"{": "__0LBRACE0",
        r"}": "__0RBRACE0__",
        r";": "__0SEMICOLON0__",
        r"@": "__0AT0__",
        r"&": "__0AMPERSAND0__",
    }

    return replacements


def encode_posix_filename(text: str) -> str:
    rs = _posix_map()
    for s, r in rs.items():
        text = text.replace(s, r)

    return text


def decode_posix_filename(text: str) -> str:
    rs = _posix_map()
    for s, r in rs.items():
        text = text.replace(r, s)

    return text
