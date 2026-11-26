#!/usr/bin/env python3
"""Query pinned Revision 1 SQLite snapshots through a read-only federation.

Python 3.6+ and SQLite 3.26+; no third-party packages are required.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCES = HERE / "FEDERATED_SOURCES.json"
QUERY_SOURCES = {
    "QL": (HERE.parent / "revision1_lineage_overlay_r4" / "queries.sql", "r4"),
    "Q": (HERE / "manifest_sidecar_r1" / "queries.sql", "sc"),
    "QX": (HERE / "dualx_exact_bridge_r1" / "EXTERNAL_EVIDENCE_BRIDGE_DRAFT.sql", "dx"),
    "QMB": (HERE / "mb15_dualx_snapshot" / "queries.sql", "mb15"),
    "QMA": (HERE / "md15_attempt_lineage_r1" / "queries.sql", "md15a"),
    "QF": (HERE / "mb15_fig5_matrix_r1" / "queries.sql", "fig5"),
    "QLR": (HERE / "learning_loop_review_r1" / "queries.sql", "learning_review"),
    "QCE": (HERE / "claim_matrix_r1" / "queries.sql", "claim_matrix"),
    "QCK": (HERE / "cross_element_checkpoint_r1" / "queries.sql", "cross_element"),
    "QLM": (HERE / "lmax1_failure_retry_r1" / "queries.sql", "lmax1"),
}
ALIASES = ("central", "r4", "sc", "dx", "mb15", "md15a", "fig5", "learning_review", "claim_matrix", "cross_element", "lmax1")


def schema_for(alias):
    return "main" if alias == "central" else alias


def quote_ident(value):
    return '"' + value.replace('"', '""') + '"'


def file_uri(path):
    # SQLite URI paths retain slash separators and percent-encode URI syntax.
    from urllib.parse import quote

    return "file:" + quote(os.path.abspath(path), safe="/") + "?mode=ro"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path):
    with open(path, "r") as handle:
        data = json.load(handle)
    sources = data.get("sources", {})
    if set(sources.keys()) != set(ALIASES):
        raise ValueError("source manifest must define exactly: " + ", ".join(ALIASES))
    return data


def check_source(alias, source, verify_hash):
    path = source["path"]
    if not os.path.isfile(path):
        raise IOError("{} database is missing or not a regular file: {}".format(alias, path))
    actual_size = os.path.getsize(path)
    if actual_size != int(source["bytes"]):
        raise ValueError("{} size mismatch: expected {}, found {}".format(alias, source["bytes"], actual_size))
    digest = None
    if verify_hash:
        digest = sha256_file(path)
        if digest.lower() != source["sha256"].lower():
            raise ValueError("{} SHA-256 mismatch: expected {}, found {}".format(alias, source["sha256"], digest))
    return {"path": path, "bytes": actual_size, "sha256_pinned": source["sha256"], "sha256_checked": digest}


def list_objects(connection, alias):
    schema = schema_for(alias)
    cursor = connection.execute(
        "SELECT name FROM {}.sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name".format(
            quote_ident(schema)
        )
    )
    return [row[0] for row in cursor.fetchall()]


def create_prefixed_views(connection):
    """Expose every source table as <alias>__<table>, preserving source IDs."""
    counts = {}
    for alias in ALIASES:
        names = list_objects(connection, alias)
        for name in names:
            view = quote_ident(alias + "__" + name)
            source = quote_ident(schema_for(alias)) + "." + quote_ident(name)
            connection.execute("CREATE TEMP VIEW {} AS SELECT * FROM {}".format(view, source))
        counts[alias] = len(names)
    return counts


def create_query_context(connection, alias):
    """Shadow unqualified names with the selected built-in query source."""
    for name in list_objects(connection, alias):
        source = quote_ident(alias) + "." + quote_ident(name)
        # SQLite resolves TEMP views before main tables for unqualified names.
        connection.execute("CREATE TEMP VIEW IF NOT EXISTS {} AS SELECT * FROM {}".format(quote_ident(name), source))


def strip_comments(text):
    # Remove SQL comments while retaining quoted string content.
    out = []
    i = 0
    quote = None
    while i < len(text):
        ch = text[i]
        if quote:
            out.append(ch)
            if ch == quote:
                if i + 1 < len(text) and text[i + 1] == quote:
                    out.append(text[i + 1])
                    i += 1
                else:
                    quote = None
        elif ch in ("'", '"', "`"):
            quote = ch
            out.append(ch)
        elif ch == "-" and i + 1 < len(text) and text[i + 1] == "-":
            while i < len(text) and text[i] != "\n":
                i += 1
            out.append("\n")
            continue
        elif ch == "/" and i + 1 < len(text) and text[i + 1] == "*":
            i += 2
            while i + 1 < len(text) and text[i : i + 2] != "*/":
                i += 1
            i += 1
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def statements(text):
    """Split SQLite statements without splitting on semicolons in literals."""
    pending = []
    result = []
    for line in text.splitlines(True):
        pending.append(line)
        candidate = "".join(pending)
        if sqlite3.complete_statement(candidate):
            cleaned = strip_comments(candidate).strip()
            if cleaned and cleaned != ";":
                result.append(cleaned.rstrip("; \t\r\n"))
            pending = []
    tail = strip_comments("".join(pending)).strip()
    if tail:
        result.append(tail.rstrip("; \t\r\n"))
    return result


def builtin_sql(query_id):
    prefix = re.match(r"^(QMB|QMA|QLR|QCE|QCK|QLM|QF|QX|QL|Q)(\d+)$", query_id, re.I)
    if not prefix:
        raise ValueError("unknown query id: " + query_id)
    family = prefix.group(1).upper()
    number = int(prefix.group(2))
    source_file, alias = QUERY_SOURCES[family]
    if family == "QCE" and number == 5:
        source_file = HERE / "claim_matrix_r2" / "queries.sql"
    elif family == "QCE" and number >= 6:
        source_file = HERE / "claim_matrix_r3" / "queries.sql"
    content = source_file.read_text()
    marker = re.compile(r"^\s*--\s*{}\d+\s*:".format(re.escape(family)), re.I | re.M)
    matches = list(marker.finditer(content))
    for index, match in enumerate(matches):
        label = re.match(r"\s*--\s*([A-Za-z]+)(\d+)\s*:", match.group(0), re.I)
        if label and int(label.group(2)) == number:
            end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            start = content.find("\n", match.end())
            start = len(content) if start < 0 else start + 1
            sqls = statements(content[start:end])
            if len(sqls) != 1:
                raise ValueError("{} must resolve to exactly one SQL statement, found {}".format(query_id, len(sqls)))
            return sqls[0], alias
    raise ValueError("{} not found in {}".format(query_id, source_file))


def validate_select(sql):
    text = strip_comments(sql).lstrip()
    if not re.match(r"^(SELECT|WITH)\b", text, re.I):
        raise ValueError("only one SELECT or WITH ... SELECT statement is allowed")
    parsed = statements(text)
    if len(parsed) != 1:
        raise ValueError("exactly one SQL statement is required")


def readonly_authorizer(action, arg1, arg2, database, source):
    # Python 3.6 does not export SQLITE_FUNCTION although SQLite 3.26 emits
    # authorizer action 31. Keep the documented SQLite action numbers as fallback.
    select_action = getattr(sqlite3, "SQLITE_SELECT", 21)
    read_action = getattr(sqlite3, "SQLITE_READ", 20)
    function_action = getattr(sqlite3, "SQLITE_FUNCTION", 31)
    allowed = {select_action, read_action, function_action}
    recursive = getattr(sqlite3, "SQLITE_RECURSIVE", 33)
    allowed.add(recursive)
    if action not in allowed:
        return sqlite3.SQLITE_DENY
    if action == function_action:
        function_name = (arg2 or arg1 or "").lower()
        if function_name in ("load_extension", "writefile"):
            return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def create_connection(sources, manifest, verify_hash):
    checked = {}
    for alias in ALIASES:
        checked[alias] = check_source(alias, sources[alias], verify_hash)
    connection = sqlite3.connect(file_uri(sources["central"]["path"]), uri=True)
    connection.row_factory = sqlite3.Row
    for alias in ALIASES:
        if alias == "central":
            continue
        connection.execute("ATTACH DATABASE ? AS {}".format(alias), (file_uri(sources[alias]["path"]),))
    versions = {}
    for alias in ALIASES:
        version = connection.execute("PRAGMA {}.user_version".format(schema_for(alias))).fetchone()[0]
        expected = int(sources[alias]["user_version"])
        if version != expected:
            raise ValueError("{} SQLite user_version mismatch: expected {}, found {}".format(alias, expected, version))
        versions[alias] = version
    views = create_prefixed_views(connection)
    return connection, {"sources": checked, "user_versions": versions, "prefixed_view_counts": views,
                        "sqlite_version": sqlite3.sqlite_version, "query_only": True}


def render_rows(cursor, fmt, metadata, query_id):
    columns = [item[0] for item in cursor.description or []]
    rows = cursor.fetchall()
    if fmt == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[column] for column in columns])
        return
    result = {
        "query_id": query_id or "custom",
        "row_count": len(rows),
        "columns": columns,
        "rows": [dict(row) for row in rows],
        "connection": metadata,
    }
    json.dump(result, sys.stdout, indent=2, sort_keys=True, default=str)
    sys.stdout.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-manifest", default=str(DEFAULT_SOURCES), help="pinned SQLite source manifest")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--query-id", help="run a registered QL21-26, Q1-7, QX01-05, QMB01-07, QMA01-07, QF501-504, QLR01-03, QCE01-09, QLM01-07, or QCK01-06 query")
    group.add_argument("--sql-file", help="run one read-only SELECT/WITH query using source-prefixed views")
    parser.add_argument("--list-queries", action="store_true", help="list registered query IDs")
    parser.add_argument("--verify-hashes", action="store_true", help="hash all pinned DBs; this reads about 4.96 GB")
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    args = parser.parse_args(argv)
    if args.list_queries:
        print("QL21-QL26  R4 lineage and Table S5")
        print("Q1-Q7      manifest sidecar")
        print("QX01-QX05  DUAL-X and exact MB bridge")
        print("QMB01-07   MB 15-model status, DUAL-X coverage, and source-qualified joins")
        print("QMA01-07   MD15 per-model attempt-lineage audit and missing evidence")
        print("QF501-504  Revision 1 Figure 5 15×8 measured source cells and status counts")
        print("QLR01-03   failure-mode audit summaries and learning-loop evidence gates")
        print("QCE01-04   Revision 1 submitted claim-to-evidence inventory, gaps, and source pins")
        print("QCE05      Fig. 4 R4 summary rows to R05 group rows; no run/attempt linkage")
        print("QCE06-09   Manuscript source versions, ledger deltas, and complete PR #10 file reconciliation; not scientific validation")
        print("QCK01-06   Cross-element checkpoint inventory, historical job status, and exact-hash joins")
        print("QLM01-07   Lmax1 Tier-1 failure, explicit retry/perturbation links, missing variants, and Tier-2 gates")
        return 0
    if not args.query_id and not args.sql_file:
        parser.error("choose --query-id or --sql-file")

    manifest = read_manifest(args.sources_manifest)
    sources = manifest["sources"]
    if args.query_id:
        sql, context_alias = builtin_sql(args.query_id)
        query_id = args.query_id
    else:
        sql = Path(args.sql_file).read_text()
        context_alias = None
        query_id = None
    validate_select(sql)

    connection, metadata = create_connection(sources, manifest, args.verify_hashes)
    if context_alias:
        create_query_context(connection, context_alias)
    connection.execute("PRAGMA query_only=ON")
    connection.set_authorizer(readonly_authorizer)
    # query_only + authorizer are active before user SQL is prepared or executed.
    cursor = connection.execute(sql)
    render_rows(cursor, args.format, metadata, query_id)
    connection.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (IOError, OSError, sqlite3.Error, ValueError) as exc:
        sys.stderr.write("federated query failed: {}\n".format(exc))
        sys.exit(2)
