"""Replay a stored episode from the Logger DB — the read side of the storage layer.

Given a run_id, pull the WHOLE history back out of SQLite and narrate it exactly like
matchmaking_demo / orchestrator_demo did live: round by round, every pairing's dialogue,
choices, outcome and payoffs, who sat idle, then the final scoreboard. This touches no
engine and no LLM — it proves the normalized schema is enough to reconstruct an episode.

Run from the repo root:

    uv run python replay.py <run_id> [--db PATH] [--config] [--calls] [--notes] [--call ID [--raw]]

--db PATH selects the SQLite database to read (default experiment.db); use it to replay runs
stored in another DB (e.g. research.db or a per-run file like qwen3/16.db).

With no run_id it lists the runs in the DB and exits. Pass --config (or -c) to also show
the episode config: the prompts in full, then the roster (each agent type plus the names
actually assigned to it), then the remaining scalar knobs (prompts / agents / name pools are
stripped from that last dump to avoid repetition).

--notes prints the memory notes each agent wrote on consolidation rounds (hidden by default).

--calls adds the raw L2 log under each pairing: one compact line per HTTP call (id, agent,
phase, attempt/http_attempt, status, tokens, and a short preview — the model's response, or
the error for failed calls); failed calls are highlighted. --call ID dumps that one call in
full — the request payload and the response body. By default escaped \n is expanded into real
line breaks for readability; add --raw to keep it verbatim. ID is the #id shown by --calls.
"""

import json
import re
import sqlite3
import sys
from datetime import datetime

DB_DEFAULT = "db/experiment.db"

# All configurable prompts live in cfg["game"]; in --config we print them in a separate
# section after the header, and strip them from the config dump to avoid duplicating walls of text.
_PROMPT_KEYS = ("rules", "talk_prompt", "talk_open_prompt", "decide_prompt", "predict_prompt",
                "reflect_prompt", "notes_prompt")
# We drop these population keys from the config dump — the roster (agents), name pools and
# the identity prompt are printed in separate sections above. We keep provider: its summary
# goes in a line under the header, and the full block (base_url, timeout_s, …) is visible in the dump.
_POP_DROP = ("agents", "first_name_pool", "last_name_pool", "identity_prompt")


def _trim_ms(ts):
    """Drop fractional seconds and the tz offset from an ISO timestamp."""
    if not ts:
        return ts
    ts = re.sub(r"\.\d+", "", ts)                  # strip fractional seconds
    return re.sub(r"[+-]\d{2}:\d{2}$", "", ts)     # strip tz offset


def _roster_line(spec):
    """One roster line 'Nx <system_prompt…>'; a long prompt is truncated to a single line.

    Old runs (before the system prompt was unified) stored a persona — shown if present."""
    sp = (spec.get("system_prompt") or spec.get("persona") or "(default system prompt)").replace("\n", " ")
    if len(sp) > 80:
        sp = sp[:79] + "…"
    return f"  {spec.get('count', 1)}x {sp}"


def _roster_names(specs, ids):
    """Names assigned to each spec: slice `ids` by their `count`.

    `ids` are the run's agent_ids in build order (SELECT ... ORDER BY rowid); specs go in the
    same order (begin inserts agents in population-build order — spec by spec), so each spec
    takes the next `count` names. Returns a list of names per spec."""
    out, i = [], 0
    for spec in specs:
        cnt = spec.get("count", 1)
        out.append(ids[i:i + cnt])
        i += cnt
    return out


def _provider_line(prov):
    """One-line provider summary; model comes last."""
    return (f"provider: temp={prov['temperature']} "
            f"max_tokens={prov['max_tokens']} model={prov['model']}")


def _duration(created, finished):
    """Wall-clock length of a run, computed from its two timestamps. '—' if unfinished."""
    if not (created and finished):
        return "—"
    try:
        secs = int((datetime.fromisoformat(finished) - datetime.fromisoformat(created)).total_seconds())
    except ValueError:
        return "?"
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m}m{s}s" if h else f"{m}m{s}s" if m else f"{s}s"


def _preview(text, n=60):
    """Short one-line preview of the text: collapse whitespace/newlines, truncate to n."""
    if not text:
        return ""
    s = " ".join(text.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _expand_newlines(s):
    r"""Expand escaped \n (two characters) into real line breaks — for a readable dump."""
    return s.replace("\\n", "\n") if s else s


def _readable(s):
    r"""JSON payload for a readable dump: the message body ("content") on its own line,
    with \n and \" expanded (escaped quotes -> regular ones)."""
    if not s:
        return s
    s = s.replace('"content": "', '"content": "\\n')   # message body starts on a new line
    return _expand_newlines(s).replace('\\"', '"')      # expand \n and \"


_YELLOW, _RESET = "\033[93m", "\033[0m"


def highlight(line, *, on):
    """Wrap a line in yellow ANSI color (highlighting messages cited by the judge)."""
    return f"{_YELLOW}{line}{_RESET}" if on else line


def cited_set(evidence_json):
    """Unpack the verdict's JSON evidence into a set of (round, pair, turn)."""
    return {(e["round"], e["pair"], e["turn"]) for e in json.loads(evidence_json)}


def load_verdict(conn, run_id):
    """Read the judge's verdict; None if there isn't one or the DB is old (no table)."""
    try:
        return conn.execute(
            "SELECT emerged, explanation, evidence FROM judge_verdicts WHERE run_id=?",
            (run_id,),
        ).fetchone()
    except sqlite3.OperationalError:                  # DB created before the judge existed
        return None


def _resolve_run_id(conn, token):
    """Accept either an integer run_id or a config_hash (design hash without rounds/judge).

    Returns an int run_id or None. For a config_hash, take the earliest run of the family
    (one design can have several runs/continuations)."""
    if token.isdigit():
        row = conn.execute("SELECT run_id FROM runs WHERE run_id=?", (int(token),)).fetchone()
        if row:
            return row[0]
    row = conn.execute(
        "SELECT run_id FROM runs WHERE config_hash=? ORDER BY created_at LIMIT 1", (token,)
    ).fetchone()
    return row[0] if row else None


def list_runs(conn):
    rows = conn.execute(
        "SELECT run_id, name, config, created_at, finished_at FROM runs ORDER BY created_at"
    ).fetchall()
    if not rows:
        print("(no runs in this DB)")
        return
    toks = {rid: (pt, ct) for rid, pt, ct in conn.execute(
        """SELECT run_id, COALESCE(SUM(usage_prompt_tokens), 0), COALESCE(SUM(usage_completion_tokens), 0)
           FROM pairings GROUP BY run_id""")}
    print(f"{len(rows)} run(s):")
    for run_id, name, config, created, finished in rows:
        cfg = json.loads(config)
        n_agents = sum(a.get("count", 1) for a in cfg["population"]["agents"])
        state = "done" if finished else "unfinished"
        label = f"  {name!r}" if name else ""
        pt, ct = toks.get(run_id, (0, 0))
        print(f"  {run_id}  {_trim_ms(created)}  "
              f"{n_agents} agents, {cfg['rounds']} rounds, {pt}+{ct} tok, {_duration(created, finished)}  "
              f"[{state}]{label}")


def _render_calls(conn, run_id, r, pi, color):
    """Compact table of a pairing's raw LLM calls (one row per HTTP attempt), with header.

    The `id` column is the call's stable id (rowid) — pass it to `--call`.
    """
    rows = conn.execute(
        """SELECT rowid, agent_id, phase, turn_idx, attempt, http_attempt,
                  status, status_code, prompt_tokens, completion_tokens, response, error
           FROM llm_calls WHERE run_id=? AND round_idx=? AND pair_idx=? ORDER BY call_idx""",
        (run_id, r, pi),
    ).fetchall()
    if not rows:
        return
    header = ("id", "agent", "phase", "a/h", "turn", "status", "code", "tokens", "preview")
    table = [header]
    statuses = []
    for (cid, agent, phase, turn, att, hatt, status, code, pt, ct, response, error) in rows:
        # preview: the model's response for ok, otherwise the error text
        preview = _preview(error if status != "ok" else response)
        table.append((
            f"#{cid}", agent, phase, f"{att}/{hatt}",
            f"t{turn}" if turn is not None else "-",
            status, str(code) if code is not None else "-", f"{pt}+{ct}", preview,
        ))
        statuses.append(status)
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]

    def fmt(cells):
        return "      " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    print(f"    llm calls ({len(rows)}):")
    print(fmt(header))
    for status, cells in zip(statuses, table[1:]):
        print(highlight(fmt(cells), on=color and status != "ok"))   # failures in yellow


def dump_call(conn, run_id, spec, raw=False):
    """Full dump of ONE call: the request payload and response — by numeric id (see --calls).

    By default escaped \\n in the payload and the raw body are expanded into real line breaks
    for readability; pass raw=True (--raw) to keep them verbatim/escaped.
    """
    try:
        cid = int(spec)
    except ValueError:
        print(f"bad --call {spec!r}; expected a numeric call id (the #id column in --calls)")
        return
    row = conn.execute(
        """SELECT round_idx, pair_idx, call_idx, agent_id, phase, turn_idx, attempt, http_attempt,
                  status, status_code, request, response, response_raw, error,
                  prompt_tokens, completion_tokens
           FROM llm_calls WHERE rowid=? AND run_id=?""",
        (cid, run_id),
    ).fetchone()
    if row is None:
        print(f"call #{cid} not found in run {run_id}.")
        return
    (r, p, ci, agent, phase, turn, att, hatt, status, code,
     request, response, raw_body, err, pt, ct) = row
    expand = (lambda s: s) if raw else _readable   # by default: body on a new line + expanded \n
    bar = "=" * 64
    print(f"{bar}\n  CALL #{cid}  (r{r}.p{p}.c{ci})   run={run_id}\n{bar}")
    print(f"agent={agent}  phase={phase}  turn_idx={turn}  attempt={att}  http_attempt={hatt}")
    print(f"status={status}  status_code={code}  tokens={pt}+{ct}" + (f"  error={err}" if err else ""))
    print("\n--- request (sent payload) ---")
    try:
        req = json.dumps(json.loads(request), indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        req = request
    print(expand(req))
    print("\n--- response (extracted text) ---")
    print(response if response is not None else "(none)")
    print("\n--- response_raw (verbatim body) ---")
    print(expand(raw_body) if raw_body is not None else "(none)")


def replay(conn, run_id, show_config=False, show_calls=False, show_notes=False):
    run = conn.execute(
        "SELECT name, config, seed, created_at, finished_at FROM runs WHERE run_id=?", (run_id,)
    ).fetchone()
    if run is None:
        print(f"run_id {run_id!r} not found in this DB.")
        list_runs(conn)
        return
    name, config, seed, created, finished = run
    cfg = json.loads(config)
    verdict = load_verdict(conn, run_id)
    cited = cited_set(verdict[2]) if verdict else set()
    color = sys.stdout.isatty()                       # ANSI only in a terminal

    pop = cfg["population"]
    n_agents = sum(a.get("count", 1) for a in pop["agents"])  # derived from counts
    # The provider is shared across the population; in old runs it lived on each agent — fallback.
    prov = pop.get("provider") or (pop["agents"][0].get("provider") if pop["agents"] else None)
    game_cfg = cfg.get("game", {})
    show_rationale = game_cfg.get("rationale", False)       # defaults match GameCfg
    show_reflection = game_cfg.get("reflection", False)     # what the run was configured to elicit
    # memory notes are printed only with the --notes flag (hidden by default); the lines will
    # still only appear where notes were actually consolidated (a_notes/b_notes != None).
    show_predictions = cfg.get("play_strategy", "direct") == "prediction"

    bar = "=" * 64
    title = f"{run_id}  ({name})" if name else run_id
    print(f"{bar}\n  REPLAY run_id={title}\n{bar}")
    print(f"{n_agents} agents, {cfg['rounds']} rounds, "
          f"max_talk_turns={cfg['game']['max_talk_turns']}")
    if prov:
        print(_provider_line(prov))                  # provider summary right under the header
    if show_config:                                  # timestamps — only in the verbose output (--config)
        print(f"created={_trim_ms(created)}  finished={_trim_ms(finished) or '(unfinished)'}")
    pt, ct = conn.execute(
        """SELECT COALESCE(SUM(usage_prompt_tokens), 0), COALESCE(SUM(usage_completion_tokens), 0)
           FROM pairings WHERE run_id=?""",
        (run_id,),
    ).fetchone()
    print(f"tokens: input={pt}  output={ct}  total={pt + ct}")

    if show_config:
        game = cfg.get("game", {})
        identity = pop.get("identity_prompt")   # shared across the population (old runs: absent)
        present = [k for k in _PROMPT_KEYS if k in game]
        print("\nprompts:")
        if not present and not identity:
            print("  (not recorded — run predates configurable prompts)")
        if identity:
            print("  [identity_prompt]")
            print("    " + identity.replace("\n", "\n    "))
        for key in present:
            text = game[key]
            print(f"  [{key}]")
            print("    " + (text.replace("\n", "\n    ") if text else "(empty)"))

        # roster — only in the verbose output, without the provider (it's already a line above and in the config dump).
        # Under each spec — the assigned names (the run's agent_ids in build order = rowid).
        print(f"\nroster ({n_agents} agents):")
        roster_ids = [a for (a,) in conn.execute(
            "SELECT agent_id FROM agents WHERE run_id=? ORDER BY rowid", (run_id,)
        )]
        for spec, names in zip(pop["agents"], _roster_names(pop["agents"], roster_ids)):
            print(_roster_line(spec))     # one line per type, as in the config
            if names:
                print(f"      {', '.join(names)}")

    if show_config:
        # config dump AFTER prompts + roster (shown above) and WITHOUT them: drop the prompt
        # strings, the agents list, the name pools and identity_prompt. The provider stays so
        # its full block (base_url, timeout_s, …) is visible beyond the one-line summary above.
        game = cfg.get("game", {})
        slim = dict(cfg)
        slim["game"] = {k: v for k, v in game.items() if k not in _PROMPT_KEYS}
        slim["population"] = {k: v for k, v in pop.items() if k not in _POP_DROP}
        print("\nconfig (prompts + roster shown above):")
        print("  " + json.dumps(slim, indent=2, sort_keys=True).replace("\n", "\n  "))

    rounds = [
        r for (r,) in conn.execute(
            "SELECT round_idx FROM rounds WHERE run_id=? ORDER BY round_idx", (run_id,)
        )
    ]
    has_evolution = any(
        row[1] == "died_round" for row in conn.execute("PRAGMA table_info(agents)")
    )
    for r in rounds:
        print(f"\n{'─' * 60}\n  ROUND {r}")
        if has_evolution and r >= 2:
            for aid, score in conn.execute(
                    "SELECT agent_id, final_score FROM agents "
                    "WHERE run_id=? AND died_round=? ORDER BY agent_id", (run_id, r)):
                print(f"  † {aid} died (score {(score or 0):g})")
            for aid, dec in conn.execute(
                    "SELECT agent_id, deceptive FROM agents "
                    "WHERE run_id=? AND born_round=? ORDER BY agent_id", (run_id, r)):
                print(f"  + {aid} joined the game{' (deceptive)' if dec else ''}")
        idle = [a for (a,) in conn.execute(
            "SELECT agent_id FROM idle WHERE run_id=? AND round_idx=? ORDER BY agent_id", (run_id, r)
        )]
        if idle:
            print(f"  idle (sat out): {', '.join(idle)}")

        pairings = conn.execute(
            """SELECT pair_idx, a_id, b_id, a_number, b_number, a_rationale, b_rationale,
                      a_outcome, a_payoff, b_payoff, a_predicted, b_predicted,
                      a_reflection, b_reflection, a_notes, b_notes, usage_calls, finished
               FROM pairings WHERE run_id=? AND round_idx=? ORDER BY pair_idx""",
            (run_id, r),
        ).fetchall()
        for (pi, a_id, b_id, a_num, b_num, a_rat, b_rat,
             outcome, a_pay, b_pay, a_pred, b_pred, a_refl, b_refl,
             a_notes, b_notes, calls, finished) in pairings:
            print(f"\n  {a_id} vs {b_id}  ({a_id} opens):")
            msgs = conn.execute(
                """SELECT speaker, text, ready FROM messages
                   WHERE run_id=? AND round_idx=? AND pair_idx=? ORDER BY turn_idx""",
                (run_id, r, pi),
            ).fetchall()
            if msgs:
                for ti, (speaker, text, ready) in enumerate(msgs):
                    mark = "   [finish=true]" if ready else ""   # finish=false is not shown
                    line = f"    {ti + 1}. {speaker}: {text}{mark}"
                    print(highlight(line, on=color) if (r, pi, ti) in cited else line)
            else:
                print("    (no messages exchanged)")
            if not finished:                          # aborted pairing: no result
                print(f"    (pairing aborted by LLM failure — no result)  [{calls} llm calls]")
            else:
                print(
                    f"    choices: {a_id}={a_num}, {b_id}={b_num}  ->  {outcome}   "
                    f"(payoffs {a_id}={a_pay:g}, {b_id}={b_pay:g})  [{calls} llm calls]"
                )
                if show_predictions:                        # gated by config (play_strategy)
                    print(f"    predictions: {a_id} guessed {b_id}={a_pred}, {b_id} guessed {a_id}={b_pred}")
                if show_rationale:                          # gated by config, not by NULL
                    print(f"      {a_id} reason: {a_rat}")
                    print(f"      {b_id} reason: {b_rat}")
                if show_reflection:
                    print(f"      {a_id} reflects: {a_refl}")
                    print(f"      {b_id} reflects: {b_refl}")
                if show_notes and (a_notes is not None or b_notes is not None):
                    print(f"      {a_id} notes: {a_notes}")   # notes are written only on consolidation rounds
                    print(f"      {b_id} notes: {b_notes}")
            if show_calls:                                  # raw L2 log (--calls)
                _render_calls(conn, run_id, r, pi, color)

    # final scoreboard + games-played, reconstructed from pairings
    scores = dict(conn.execute(
        "SELECT agent_id, final_score FROM agents WHERE run_id=?", (run_id,)
    ))
    games = {aid: 0 for aid in scores}
    for (a_id, b_id) in conn.execute(
        "SELECT a_id, b_id FROM pairings WHERE run_id=? AND finished=1", (run_id,)
    ):
        games[a_id] += 1
        games[b_id] += 1

    print(f"\n{bar}\n  FINAL SCOREBOARD\n{bar}")
    for aid, score in sorted(scores.items(), key=lambda kv: (kv[1] or 0), reverse=True):
        s = "?" if score is None else f"{score:g}"
        print(f"  {aid}: {s}   ({games[aid]} games)")

    dist = dict(conn.execute(
        "SELECT a_outcome, COUNT(*) FROM pairings WHERE run_id=? AND finished=1 GROUP BY a_outcome",
        (run_id,)
    ))
    total = sum(dist.values())
    cc = f"{dist.get('CC', 0) / total * 100:.0f}%" if total else "n/a"
    print(f"\noutcomes: {dist}   CC={cc}   games={total}")

    if verdict:
        emerged, explanation, evidence_json = verdict
        print(f"\n{bar}\n  JUDGE VERDICT\n{bar}")
        print(f"  reputation institute emerged: {'YES' if emerged else 'NO'}")
        print(f"  {explanation}")
        refs = json.loads(evidence_json)
        if refs:
            print(f"\n  evidence ({len(refs)} message(s)):")
            for e in refs:
                row = conn.execute(
                    """SELECT speaker, text FROM messages
                       WHERE run_id=? AND round_idx=? AND pair_idx=? AND turn_idx=?""",
                    (run_id, e["round"], e["pair"], e["turn"]),
                ).fetchone()
                if row:
                    line = f"    r{e['round']}.p{e['pair']}.t{e['turn']}  {row[0]}: {row[1]}"
                    print(highlight(line, on=color))


def main():
    args = sys.argv[1:]
    show_config = "--config" in args or "-c" in args
    show_calls = "--calls" in args
    show_notes = "--notes" in args                 # memory notes are hidden by default
    raw = "--raw" in args                          # --call: don't expand \n
    call_spec = None
    db_path = DB_DEFAULT
    pos = []
    skip = False
    for i, a in enumerate(args):
        if skip:                                  # the value after --call / --db
            skip = False
            continue
        if a == "--call":
            call_spec = args[i + 1] if i + 1 < len(args) else None
            skip = True
            continue
        if a == "--db":
            db_path = args[i + 1] if i + 1 < len(args) else db_path
            skip = True
            continue
        if a in ("--config", "-c", "--calls", "--raw", "--notes"):
            continue
        pos.append(a)                             # positional args only

    conn = sqlite3.connect(db_path)
    try:
        if not pos:
            print(f"usage: replay.py <run_id> [--db PATH] [--config] [--calls] [--notes] [--call ID [--raw]]   "
                  f"(run_id — a number or config_hash; db: {db_path})\n")
            list_runs(conn)
        else:
            target = _resolve_run_id(conn, pos[0])         # a number or a legacy hash -> int run_id
            if target is None:
                print(f"run {pos[0]!r} not found in this DB.\n")
                list_runs(conn)
            elif call_spec is not None:
                dump_call(conn, target, call_spec, raw=raw)    # full dump of one call
            else:
                replay(conn, target, show_config=show_config, show_calls=show_calls,
                       show_notes=show_notes)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
