"""Version-stable rendering of signatures for the API reference.

`str(inspect.signature(...))` is not stable across versions: pandas 2 renders a
Series annotation as ``pandas.core.series.Series`` and pandas 3 renders the same
annotation as ``pandas.Series``, and from Python 3.14 ``Optional[int]`` reprs as
``int | None``.  Generating ``docs/api.md`` from one of those and testing it
against another is a guaranteed CI failure on some job.

So the page and its test both render signatures through the functions here,
which strip module qualifiers and put unions into a single form.  There is one
renderer, imported from one place, so the page cannot disagree with the test.
"""

import inspect
import re
import types
import typing

__all__ = ["render_annotation", "render_signature", "markdown_signature"]

_NoneType = type(None)

# ``pandas.core.series.Series`` -> ``Series``.  Requires a dot, so string
# literals inside e.g. ``Literal['a', 'b']`` are untouched.
_QUALIFIER = re.compile(r"\b(?:[A-Za-z_][A-Za-z0-9_]*\.)+([A-Za-z_][A-Za-z0-9_]*)\b")


def _split_top_level(text, sep="|"):
    parts, depth, current = [], 0, ""
    for char in text:
        if char in "[({":
            depth += 1
        elif char in "])}":
            depth -= 1
        if char == sep and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    parts.append(current)
    return [part.strip() for part in parts]


def _union(rendered):
    """One spelling for a union, whatever the source spelling was."""
    non_none = [part for part in rendered if part not in ("None", "NoneType")]
    nones = len(rendered) - len(non_none)
    if nones and len(non_none) == 1:
        return f"Optional[{non_none[0]}]"
    if nones:
        return "Union[" + ", ".join(non_none + ["NoneType"]) + "]"
    return "Union[" + ", ".join(non_none) + "]"


def _clean_text(text):
    """Normalise an annotation that reaches us as a string.

    ``from __future__ import annotations`` in pandas means many of its
    signatures carry string annotations such as ``'int | None'``.
    """
    text = text.strip()
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    text = _QUALIFIER.sub(r"\1", text)
    if "|" in text:
        parts = _split_top_level(text)
        if len(parts) > 1:
            return _union([_clean_text(part) for part in parts])
    return text


def render_annotation(annotation):
    """Render an annotation identically under any supported pandas/Python."""
    if isinstance(annotation, str):
        return _clean_text(annotation)
    if isinstance(annotation, typing.ForwardRef):
        return _clean_text(annotation.__forward_arg__)
    if annotation is _NoneType:
        return "NoneType"

    origin = typing.get_origin(annotation)
    if origin is not None:
        args = typing.get_args(annotation)
        if origin is typing.Union or origin is getattr(types, "UnionType", ()):
            return _union([render_annotation(arg) for arg in args])
        name = getattr(origin, "__name__", None) or _clean_text(repr(origin))
        if not args:
            return name
        return f"{name}[{', '.join(render_annotation(arg) for arg in args)}]"

    if isinstance(annotation, type):
        return annotation.__name__
    return _clean_text(repr(annotation))


def render_signature(func):
    """Render ``func``'s signature, without ``self``, version-stably.

    Parameter names, kinds and defaults come straight from
    ``inspect.signature``; only the annotation text is normalised.
    """
    signature = inspect.signature(func)
    parts, previous_kind = [], None
    for parameter in signature.parameters.values():
        if parameter.name == "self" and not parts:
            previous_kind = parameter.kind
            continue
        if (
            previous_kind is inspect.Parameter.POSITIONAL_ONLY
            and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY
        ):
            parts.append("/")
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY and previous_kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            parts.append("*")

        rendered = parameter.name
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            rendered = "*" + rendered
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            rendered = "**" + rendered
        if parameter.annotation is not inspect.Parameter.empty:
            rendered += f": {render_annotation(parameter.annotation)}"
            if parameter.default is not inspect.Parameter.empty:
                rendered += f" = {parameter.default!r}"
        elif parameter.default is not inspect.Parameter.empty:
            rendered += f"={parameter.default!r}"
        parts.append(rendered)
        previous_kind = parameter.kind

    if previous_kind is inspect.Parameter.POSITIONAL_ONLY:
        parts.append("/")

    text = "(" + ", ".join(parts) + ")"
    if signature.return_annotation is not inspect.Signature.empty:
        text += f" -> {render_annotation(signature.return_annotation)}"
    return text


def markdown_signature(func):
    """``render_signature`` escaped for a markdown table cell."""
    return render_signature(func).replace("|", r"\|")
