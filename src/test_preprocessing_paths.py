"""Regression checks for path/query splitting in parse_http_string."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import parse_http_string


def test_path_with_query_splits():
    row = parse_http_string("/api/workspaces/insky/user-favorites/?all=true")
    assert row["path"] == "/api/workspaces/insky/user-favorites/"
    assert row["query"] == "all=true"


def test_bare_query_stays_query():
    row = parse_http_string("id=1&name=test")
    assert row["path"] == ""
    assert row["query"] == "id=1&name=test"


def test_path_only():
    row = parse_http_string("/api/workspaces/insky/modules/")
    assert row["path"] == "/api/workspaces/insky/modules/"
    assert row["query"] == ""


def test_patch_method_splits_path():
    row = parse_http_string(
        "PATCH /api/workspaces/insky/projects/bce0a79c-90d2-4558-9084-945ad6acbdae/issues/9906eeae-3678-40e2-9869-64bc8b84c7c5/"
    )
    assert row["method"] == "PATCH"
    assert row["path"] == "/api/workspaces/insky/projects/bce0a79c-90d2-4558-9084-945ad6acbdae/issues/9906eeae-3678-40e2-9869-64bc8b84c7c5/"
    assert "PATCH" not in row["path"]


if __name__ == "__main__":
    test_path_with_query_splits()
    test_bare_query_stays_query()
    test_path_only()
    test_patch_method_splits_path()
    print("ok")
