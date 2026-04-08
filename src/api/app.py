from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.retrieval.search import load_index, resolve_index_dir, search_index


APP_TITLE = "Industrial Electrical Fault Diagnosis API"
APP_VERSION = "0.3.0"

tags_metadata = [
    {"name": "system", "description": "服务状态、索引信息、示例入口。"},
    {"name": "search", "description": "知识块检索接口。"},
    {"name": "ask", "description": "基于检索结果生成简要回答。"},
    {"name": "debug", "description": "更适合人工使用的调试页面。"},
]


def get_default_config_path() -> str:
    return os.getenv("MVP_CONFIG_PATH", "D:/MVP_Project/config/chunking_config.yaml")


def get_default_index_dir() -> str | None:
    value = os.getenv("MVP_INDEX_DIR")
    return value if value else None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户查询")
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数量")
    doc_type: str | None = Field(default=None, description="按文档类型过滤，如 fault_page")
    doc_role: str | None = Field(default=None, description="按文档角色过滤，如 troubleshooting")
    event_code: str | None = Field(default=None, description="按故障/报警码过滤，如 F2 / A0501")
    param_id: str | None = Field(default=None, description="按参数号过滤，如 P0010")
    protocol: str | None = Field(default=None, description="按协议过滤，如 MODBUS / USS")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "V20 报 F2 是什么意思",
                    "top_k": 5,
                    "doc_type": "fault_page",
                    "doc_role": None,
                    "event_code": None,
                    "param_id": None,
                    "protocol": None,
                }
            ]
        }
    }


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户问题")
    top_k: int = Field(default=5, ge=1, le=10, description="检索返回数量")
    doc_type: str | None = Field(default=None, description="按文档类型过滤")
    doc_role: str | None = Field(default=None, description="按文档角色过滤")
    event_code: str | None = Field(default=None, description="按故障/报警码过滤")
    param_id: str | None = Field(default=None, description="按参数号过滤")
    protocol: str | None = Field(default=None, description="按协议过滤")
    max_context_results: int = Field(default=3, ge=1, le=5, description="回答时最多参考前几条结果")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "V20 报 F2 是什么意思，先检查什么？",
                    "top_k": 5,
                    "doc_type": None,
                    "doc_role": None,
                    "event_code": None,
                    "param_id": None,
                    "protocol": None,
                    "max_context_results": 3,
                }
            ]
        }
    }


class CitationModel(BaseModel):
    doc_title: str = ""
    doc_type: str = ""
    page_start: int | None = None
    page_end: int | None = None
    section: str = ""


class SearchResultModel(BaseModel):
    rank: int
    score: float
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    citation: CitationModel


class SearchResponse(BaseModel):
    query: str
    entities: dict[str, str | None] | None = None
    filters: dict[str, Any] | None = None
    index_dir: str
    build_info: dict[str, Any] | None = None
    results: list[SearchResultModel]


class AskResponse(BaseModel):
    query: str
    answer: str
    answer_mode: str
    entities: dict[str, str | None] | None = None
    filters: dict[str, Any] | None = None
    index_dir: str
    citations: list[CitationModel]
    results: list[SearchResultModel]


class HealthResponse(BaseModel):
    status: str
    config_path: str
    index_dir: str
    index_loaded: bool
    num_records: int | None = None
    num_features: int | None = None


class IndexInfoResponse(BaseModel):
    index_dir: str
    build_info: dict[str, Any]
    num_records: int
    num_features: int


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=(
        "用于调试工业电气故障诊断知识库的本地检索服务。\n\n"
        "建议使用顺序：\n"
        "1. 先访问 `/` 查看导航\n"
        "2. 再访问 `/playground` 做人工调试\n"
        "3. 最后再去 `/docs` 看接口细节"
    ),
    openapi_tags=tags_metadata,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_filters_from_search_request(req: SearchRequest) -> dict[str, Any]:
    return {
        "doc_type": req.doc_type,
        "doc_role": req.doc_role,
        "event_code": req.event_code,
        "param_id": req.param_id,
        "protocol": req.protocol,
    }


def build_filters_from_ask_request(req: AskRequest) -> dict[str, Any]:
    return {
        "doc_type": req.doc_type,
        "doc_role": req.doc_role,
        "event_code": req.event_code,
        "param_id": req.param_id,
        "protocol": req.protocol,
    }


def get_runtime_paths() -> tuple[str, Path]:
    config_path = get_default_config_path()
    index_dir = resolve_index_dir(
        config_path=config_path,
        explicit_index_dir=get_default_index_dir(),
    )
    return config_path, index_dir


def clean_text_preview(text: str, max_chars: int = 260) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text[:max_chars].rstrip() + ("..." if len(text) > max_chars else "")


def dedupe_citations(results: list[dict[str, Any]], max_items: int = 3) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    citations: list[dict[str, Any]] = []

    for item in results:
        citation = item.get("citation", {}) or {}
        key = (
            citation.get("doc_title"),
            citation.get("page_start"),
            citation.get("page_end"),
            citation.get("section"),
        )
        if key in seen:
            continue
        seen.add(key)
        citations.append(citation)
        if len(citations) >= max_items:
            break

    return citations


def build_answer_from_results(
    query: str,
    search_output: dict[str, Any],
    max_context_results: int = 3,
) -> tuple[str, list[dict[str, Any]]]:
    results = search_output.get("results", []) or []
    entities = search_output.get("entities", {}) or {}

    if not results:
        answer = (
            "未在当前知识库中检索到足够相关的内容。"
            "建议你换一种问法，或补充故障码、参数号、协议名等关键信息。"
        )
        return answer, []

    top_results = results[:max_context_results]
    citations = dedupe_citations(top_results, max_items=max_context_results)
    top1 = top_results[0]
    top1_text = clean_text_preview(top1.get("text", ""), 280)

    event_code = entities.get("event_code")
    param_id = entities.get("param_id")
    macro_id = entities.get("macro_id")
    protocol = entities.get("protocol")

    intro = "根据当前知识库检索结果，"
    if event_code:
        intro += f"与 {event_code} 相关的资料显示："
    elif param_id:
        intro += f"与参数 {param_id} 相关的资料显示："
    elif macro_id:
        intro += f"与宏 {macro_id} 相关的资料显示："
    elif protocol:
        intro += f"与 {protocol} 通讯相关的资料显示："
    else:
        intro += "相关资料显示："

    lines: list[str] = [intro, f"1. 最高相关内容：{top1_text}"]

    if len(top_results) > 1:
        for idx, item in enumerate(top_results[1:], start=2):
            snippet = clean_text_preview(item.get("text", ""), 180)
            lines.append(f"{idx}. 补充参考：{snippet}")

    lines.append("请优先查看下方引用来源中的原始片段与页码。")
    answer = "\n".join(lines)
    return answer, citations


@app.get("/", response_class=HTMLResponse, tags=["debug"], summary="首页导航")
def root() -> str:
    config_path, index_dir = get_runtime_paths()

    return f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>{APP_TITLE}</title>
        <style>
            body {{
                font-family: Arial, "Microsoft YaHei", sans-serif;
                max-width: 900px;
                margin: 40px auto;
                padding: 0 20px;
                line-height: 1.6;
                color: #222;
            }}
            h1 {{
                margin-bottom: 8px;
            }}
            .muted {{
                color: #666;
                margin-bottom: 24px;
            }}
            .card {{
                border: 1px solid #ddd;
                border-radius: 12px;
                padding: 18px 20px;
                margin-bottom: 16px;
                background: #fafafa;
            }}
            .btns a {{
                display: inline-block;
                margin: 6px 10px 6px 0;
                padding: 10px 14px;
                text-decoration: none;
                border-radius: 8px;
                border: 1px solid #ccc;
                color: #111;
                background: white;
            }}
            code {{
                background: #f0f0f0;
                padding: 2px 6px;
                border-radius: 6px;
            }}
            ul {{
                padding-left: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>{APP_TITLE}</h1>
        <div class="muted">版本：{APP_VERSION}</div>

        <div class="card">
            <strong>推荐使用顺序</strong>
            <ol>
                <li>先点 <a href="/health">/health</a> 确认索引能正常加载</li>
                <li>再点 <a href="/playground">/playground</a> 直接输入问题调试</li>
                <li>最后再去 <a href="/docs">/docs</a> 查看完整接口说明</li>
            </ol>
        </div>

        <div class="card btns">
            <strong>快捷入口</strong><br />
            <a href="/health">健康检查</a>
            <a href="/index-info">索引信息</a>
            <a href="/sample-queries">示例查询</a>
            <a href="/playground">调试页面</a>
            <a href="/docs">Swagger 文档</a>
            <a href="/redoc">ReDoc 文档</a>
        </div>

        <div class="card">
            <strong>当前运行配置</strong>
            <ul>
                <li>config_path: <code>{config_path}</code></li>
                <li>index_dir: <code>{str(index_dir)}</code></li>
            </ul>
        </div>

        <div class="card">
            <strong>建议先试的问题</strong>
            <ul>
                <li>V20 报 F2 是什么意思</li>
                <li>A0501 怎么处理</li>
                <li>P0010 是什么参数</li>
                <li>Modbus 怎么读写参数</li>
            </ul>
        </div>
    </body>
    </html>
    """


@app.get("/playground", response_class=HTMLResponse, tags=["debug"], summary="浏览器调试页")
def playground() -> str:
    return """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Ask / Search Playground</title>
        <style>
            body {
                font-family: Arial, "Microsoft YaHei", sans-serif;
                max-width: 1100px;
                margin: 30px auto;
                padding: 0 20px;
                color: #222;
            }
            h1 {
                margin-bottom: 8px;
            }
            .muted {
                color: #666;
                margin-bottom: 18px;
            }
            .panel {
                border: 1px solid #ddd;
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 18px;
                background: #fafafa;
            }
            label {
                display: block;
                margin-top: 10px;
                margin-bottom: 6px;
                font-weight: 600;
            }
            input, textarea, select, button {
                width: 100%;
                box-sizing: border-box;
                padding: 10px 12px;
                border: 1px solid #ccc;
                border-radius: 8px;
                font-size: 14px;
            }
            .row {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 12px;
            }
            .row-2 {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
            }
            .btn-row {
                display: flex;
                gap: 10px;
                margin-top: 14px;
            }
            .btn-row button {
                width: auto;
                min-width: 120px;
                cursor: pointer;
                background: white;
            }
            pre {
                white-space: pre-wrap;
                word-break: break-word;
                background: #111;
                color: #f5f5f5;
                padding: 14px;
                border-radius: 10px;
                overflow: auto;
            }
            .answer-box {
                white-space: pre-wrap;
                word-break: break-word;
                background: white;
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 14px;
                min-height: 80px;
            }
            .small {
                color: #666;
                font-size: 13px;
            }
        </style>
    </head>
    <body>
        <h1>检索 / 问答 调试页面</h1>
        <div class="muted">可以直接在浏览器里测试 /search 和 /ask。</div>

        <div class="panel">
            <label for="mode">模式</label>
            <select id="mode">
                <option value="ask" selected>问答（/ask）</option>
                <option value="search">检索（/search）</option>
            </select>

            <label for="query">问题</label>
            <textarea id="query" rows="3">V20 报 F2 是什么意思，先检查什么？</textarea>

            <div class="row">
                <div>
                    <label for="top_k">top_k</label>
                    <input id="top_k" type="number" value="5" min="1" max="20" />
                </div>
                <div>
                    <label for="doc_type">doc_type</label>
                    <input id="doc_type" type="text" placeholder="例如 fault_page" />
                </div>
                <div>
                    <label for="doc_role">doc_role</label>
                    <input id="doc_role" type="text" placeholder="例如 troubleshooting" />
                </div>
            </div>

            <div class="row">
                <div>
                    <label for="event_code">event_code</label>
                    <input id="event_code" type="text" placeholder="例如 F2" />
                </div>
                <div>
                    <label for="param_id">param_id</label>
                    <input id="param_id" type="text" placeholder="例如 P0010" />
                </div>
                <div>
                    <label for="protocol">protocol</label>
                    <input id="protocol" type="text" placeholder="例如 MODBUS" />
                </div>
            </div>

            <div class="row-2">
                <div>
                    <label for="max_context_results">max_context_results（仅 /ask 使用）</label>
                    <input id="max_context_results" type="number" value="3" min="1" max="5" />
                </div>
                <div>
                    <label>&nbsp;</label>
                    <div class="small">/search 模式会忽略这个字段</div>
                </div>
            </div>

            <div class="btn-row">
                <button onclick="runRequest()">执行</button>
                <button onclick="loadFaultExample()">故障示例</button>
                <button onclick="loadModbusExample()">通讯示例</button>
            </div>
        </div>

        <div class="panel">
            <strong>回答（仅 /ask）</strong>
            <div id="answer" class="answer-box">执行 /ask 后，这里会显示简要回答。</div>
        </div>

        <div class="panel">
            <strong>原始 JSON 返回</strong>
            <pre id="output">点击“执行”后，这里会显示返回结果。</pre>
        </div>

        <script>
            function loadFaultExample() {
                document.getElementById("mode").value = "ask";
                document.getElementById("query").value = "V20 报 F2 是什么意思，先检查什么？";
                document.getElementById("top_k").value = "5";
                document.getElementById("doc_type").value = "";
                document.getElementById("doc_role").value = "";
                document.getElementById("event_code").value = "";
                document.getElementById("param_id").value = "";
                document.getElementById("protocol").value = "";
                document.getElementById("max_context_results").value = "3";
            }

            function loadModbusExample() {
                document.getElementById("mode").value = "ask";
                document.getElementById("query").value = "Modbus 怎么读写参数";
                document.getElementById("top_k").value = "5";
                document.getElementById("doc_type").value = "";
                document.getElementById("doc_role").value = "";
                document.getElementById("event_code").value = "";
                document.getElementById("param_id").value = "";
                document.getElementById("protocol").value = "MODBUS";
                document.getElementById("max_context_results").value = "3";
            }

            async function runRequest() {
                const mode = document.getElementById("mode").value;
                const payload = {
                    query: document.getElementById("query").value,
                    top_k: Number(document.getElementById("top_k").value || 5),
                    doc_type: document.getElementById("doc_type").value || null,
                    doc_role: document.getElementById("doc_role").value || null,
                    event_code: document.getElementById("event_code").value || null,
                    param_id: document.getElementById("param_id").value || null,
                    protocol: document.getElementById("protocol").value || null,
                    max_context_results: Number(document.getElementById("max_context_results").value || 3)
                };

                const output = document.getElementById("output");
                const answer = document.getElementById("answer");
                output.textContent = "请求中...";
                answer.textContent = "请求中...";

                try {
                    const resp = await fetch(`/${mode}`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload)
                    });

                    const data = await resp.json();
                    output.textContent = JSON.stringify(data, null, 2);

                    if (mode === "ask") {
                        answer.textContent = data.answer || "未返回 answer 字段。";
                    } else {
                        answer.textContent = "当前是 /search 模式，不生成 answer。";
                    }
                } catch (err) {
                    output.textContent = "请求失败: " + String(err);
                    answer.textContent = "请求失败。";
                }
            }
        </script>
    </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse, tags=["system"], summary="健康检查")
def health_check() -> HealthResponse:
    config_path, index_dir = get_runtime_paths()

    try:
        vectorizer, _matrix, records, _build_info = load_index(index_dir)
        num_features = len(getattr(vectorizer, "vocabulary_", {}) or {})
        return HealthResponse(
            status="ok",
            config_path=config_path,
            index_dir=str(index_dir),
            index_loaded=True,
            num_records=len(records),
            num_features=num_features,
        )
    except Exception as e:
        return HealthResponse(
            status=f"error: {e}",
            config_path=config_path,
            index_dir=str(index_dir),
            index_loaded=False,
            num_records=None,
            num_features=None,
        )


@app.get("/index-info", response_model=IndexInfoResponse, tags=["system"], summary="查看索引信息")
def index_info() -> IndexInfoResponse:
    _config_path, index_dir = get_runtime_paths()

    try:
        vectorizer, _matrix, records, build_info = load_index(index_dir)
        num_features = len(getattr(vectorizer, "vocabulary_", {}) or {})
        return IndexInfoResponse(
            index_dir=str(index_dir),
            build_info=build_info,
            num_records=len(records),
            num_features=num_features,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load index: {e}") from e


@app.post(
    "/search",
    response_model=SearchResponse,
    tags=["search"],
    summary="检索知识块",
    description="主检索接口。建议先从这个接口开始调试。",
)
def search_api(req: SearchRequest) -> SearchResponse:
    _config_path, index_dir = get_runtime_paths()
    filters = build_filters_from_search_request(req)

    try:
        result = search_index(
            query=req.query,
            index_dir=index_dir,
            top_k=req.top_k,
            filters=filters,
        )
        return SearchResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Index not found: {e}") from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}") from e


@app.get(
    "/search",
    response_model=SearchResponse,
    tags=["search"],
    summary="GET 方式检索知识块",
    description="方便浏览器地址栏直接调试。",
)
def search_api_get(
    query: str = Query(..., min_length=1, description="用户查询"),
    top_k: int = Query(default=5, ge=1, le=20, description="返回结果数量"),
    doc_type: str | None = Query(default=None, description="文档类型过滤"),
    doc_role: str | None = Query(default=None, description="文档角色过滤"),
    event_code: str | None = Query(default=None, description="故障码过滤"),
    param_id: str | None = Query(default=None, description="参数号过滤"),
    protocol: str | None = Query(default=None, description="协议过滤"),
) -> SearchResponse:
    req = SearchRequest(
        query=query,
        top_k=top_k,
        doc_type=doc_type,
        doc_role=doc_role,
        event_code=event_code,
        param_id=param_id,
        protocol=protocol,
    )
    return search_api(req)


@app.post(
    "/ask",
    response_model=AskResponse,
    tags=["ask"],
    summary="基于检索结果生成简要回答",
    description="当前为本地规则版问答：先检索，再基于前几条结果做摘要回答，并返回引用。",
)
def ask_api(req: AskRequest) -> AskResponse:
    _config_path, index_dir = get_runtime_paths()
    filters = build_filters_from_ask_request(req)

    try:
        search_output = search_index(
            query=req.query,
            index_dir=index_dir,
            top_k=req.top_k,
            filters=filters,
        )
        answer, citations = build_answer_from_results(
            query=req.query,
            search_output=search_output,
            max_context_results=req.max_context_results,
        )
        return AskResponse(
            query=req.query,
            answer=answer,
            answer_mode="retrieval_summary",
            entities=search_output.get("entities"),
            filters=search_output.get("filters"),
            index_dir=search_output.get("index_dir", str(index_dir)),
            citations=[CitationModel(**c) for c in citations],
            results=[SearchResultModel(**r) for r in search_output.get("results", [])],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Index not found: {e}") from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ask failed: {e}") from e


@app.get(
    "/ask",
    response_model=AskResponse,
    tags=["ask"],
    summary="GET 方式问答",
    description="方便浏览器地址栏直接调试 /ask。",
)
def ask_api_get(
    query: str = Query(..., min_length=1, description="用户问题"),
    top_k: int = Query(default=5, ge=1, le=10, description="检索返回数量"),
    doc_type: str | None = Query(default=None, description="文档类型过滤"),
    doc_role: str | None = Query(default=None, description="文档角色过滤"),
    event_code: str | None = Query(default=None, description="故障码过滤"),
    param_id: str | None = Query(default=None, description="参数号过滤"),
    protocol: str | None = Query(default=None, description="协议过滤"),
    max_context_results: int = Query(default=3, ge=1, le=5, description="回答时最多参考前几条结果"),
) -> AskResponse:
    req = AskRequest(
        query=query,
        top_k=top_k,
        doc_type=doc_type,
        doc_role=doc_role,
        event_code=event_code,
        param_id=param_id,
        protocol=protocol,
        max_context_results=max_context_results,
    )
    return ask_api(req)


@app.get("/sample-queries", tags=["system"], summary="示例查询")
def sample_queries() -> dict[str, list[str]]:
    return {
        "examples": [
            "V20 报 F2 是什么意思",
            "A0501 怎么处理",
            "P0010 是什么参数",
            "Cn001 是什么意思",
            "Modbus 怎么读写参数",
        ]
    }

# python -m uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
# http://127.0.0.1:8000/
# http://127.0.0.1:8000/playground
# http://127.0.0.1:8000/docs
