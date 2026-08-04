"""主图与 supervisor 子图的编译结构测试。

验证 graph.py 模块级编译产物的节点、边与路由契约，
不触真实 LLM（纯结构断言，秒级完成）。
"""

from __future__ import annotations

from company_report_kit.graph.graph import graph, supervisor_subgraph


# ──────────────────────────────────────────────────────────────
# supervisor 子图
# ──────────────────────────────────────────────────────────────


def test_supervisor_subgraph_has_expected_nodes() -> None:
    """supervisor 子图仅含 supervisor + supervisor_tools 两节点。"""
    node_names = set(supervisor_subgraph.nodes.keys()) - {"__start__"}
    assert node_names == {"supervisor", "supervisor_tools"}


def test_supervisor_subgraph_starts_at_supervisor() -> None:
    """子图入口固定为 supervisor 节点。"""
    edges = supervisor_subgraph.get_graph().edges
    start_edges = [e for e in edges if e.source == "__start__"]
    assert len(start_edges) == 1
    assert start_edges[0].target == "supervisor"


def test_supervisor_subgraph_supervisor_routes_to_tools() -> None:
    """supervisor 节点有条件边指向 supervisor_tools（Command goto 路由）。"""
    edges = supervisor_subgraph.get_graph().edges
    route_edges = [e for e in edges if e.source == "supervisor"]
    assert any(e.target == "supervisor_tools" and e.conditional for e in route_edges)


def test_supervisor_subgraph_tools_can_loop_and_exit() -> None:
    """supervisor_tools 有回环边（→supervisor）和退出边（→__end__），均由 Command 路由。"""
    edges = supervisor_subgraph.get_graph().edges
    tool_edges = [e for e in edges if e.source == "supervisor_tools"]
    targets = {e.target for e in tool_edges}
    assert "supervisor" in targets  # 回环继续
    assert "__end__" in targets  # 退出
    assert all(e.conditional for e in tool_edges)  # 全部条件边


# ──────────────────────────────────────────────────────────────
# 主图
# ──────────────────────────────────────────────────────────────


def test_main_graph_has_expected_nodes() -> None:
    """主图含四个业务节点（不含 __start__）。"""
    node_names = set(graph.nodes.keys()) - {"__start__"}
    assert node_names == {
        "clarify_with_user",
        "write_brief",
        "research_supervisor",
        "final_report_generation",
    }


def test_main_graph_starts_at_clarify() -> None:
    """主图入口固定为 clarify_with_user。"""
    edges = graph.get_graph().edges
    start_edges = [e for e in edges if e.source == "__start__"]
    assert len(start_edges) == 1
    assert start_edges[0].target == "clarify_with_user"
    assert not start_edges[0].conditional  # 固定边，非条件路由


def test_main_graph_research_to_final_report_is_fixed() -> None:
    """research_supervisor → final_report_generation → __end__ 是固定流水线。"""
    edges = graph.get_graph().edges
    # research_supervisor → final_report_generation
    r_edges = [e for e in edges if e.source == "research_supervisor"]
    assert len(r_edges) == 1
    assert r_edges[0].target == "final_report_generation"
    assert not r_edges[0].conditional
    # final_report_generation → __end__
    f_edges = [e for e in edges if e.source == "final_report_generation"]
    assert len(f_edges) == 1
    assert f_edges[0].target == "__end__"


def test_main_graph_clarify_has_conditional_branches() -> None:
    """clarify_with_user 有两条条件边：interrupt(__end__) 与 write_brief。"""
    edges = graph.get_graph().edges
    c_edges = [e for e in edges if e.source == "clarify_with_user" and e.conditional]
    targets = {e.target for e in c_edges}
    assert "__end__" in targets  # 需追问时 interrupt
    assert "write_brief" in targets  # 无需追问时放行


def test_main_graph_write_brief_routes_to_research() -> None:
    """write_brief 有条件边指向 research_supervisor（Command 路由放行后进入）。"""
    edges = graph.get_graph().edges
    wb_edges = [e for e in edges if e.source == "write_brief"]
    assert any(e.target == "research_supervisor" for e in wb_edges)
