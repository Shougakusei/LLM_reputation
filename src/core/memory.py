from __future__ import annotations

from dataclasses import dataclass

from src.core.config import GameCfg
from src.providers.base import Message

# Fallback set of transcript templates if render() wasn't given a GameCfg (direct render()
# calls in tests, phases without a game config). Matches the GameCfg defaults.
_DEFAULT_GAME = GameCfg()


@dataclass
class MemoryEntry:
    round: int
    my_id: str              # the agent's own id (for the "<my_id> (you)" label in the diary)
    partner_id: str
    transcript: list[dict]  # [{speaker, text, ready}]
    my_number: int
    my_rationale: str
    partner_number: int
    outcome: str
    payoff: float            # the agent's own payoff this round
    partner_payoff: float    # the partner's payoff (for the symmetric "Payoffs: ..." line)
    score: float = 0.0       # the agent's accumulated score BEFORE this round (same as in the phase header)
    my_predicted: int | None = None  # prediction strategy; None for direct
    my_reflection: str | None = None  # post-game reflection; None if disabled


class Memory:
    def __init__(self) -> None:
        self.entries: list[MemoryEntry] = []
        self.notes: str | None = None   # compressed memory (memory notes); None = no notes yet
        self.noted_upto: int = 0         # how many entries are already folded into notes (buffer boundary)

    def add(self, entry: MemoryEntry) -> None:
        self.entries.append(entry)

    def set_notes(self, notes: str) -> None:
        """Remember the fresh notes and shift the boundary: everything played so far is folded in."""
        self.notes = notes
        self.noted_upto = len(self.entries)

    def fragments(self, window: int | None, cfg: GameCfg | None = None) -> dict[str, str]:
        """The memory materials a step prompt can lay out, one string per placeholder.

        Keys mirror the step-prompt placeholders: `history_lines` — rounds already folded
        into the note, one `history_line` each (empty when the template is unset);
        `recent_rounds` — rounds not yet folded, rendered in full via `history_prompt`
        (with notes off that is the whole past, the window applies); `notes` — the raw
        note text; `notes_line` — the note rendered through `msg_self`.
        """
        cfg = cfg or _DEFAULT_GAME
        folded = self.entries[: self.noted_upto]
        recent = _window(self.entries[self.noted_upto:], window)
        notes = self.notes or ""
        return {
            "history_lines": ("\n".join(_render_entry(e, cfg, cfg.history_line) for e in folded)
                              if cfg.history_line else ""),
            "recent_rounds": "\n".join(_render_entry(e, cfg) for e in recent),
            "notes_line": cfg.msg_self.replace("{text}", notes) if notes else "",
            "notes": notes,
        }

    def render(self, window: int | None, cfg: GameCfg | None = None) -> list[Message]:
        # Past rounds are rendered as one game transcript (tags <game>/<you>/<name>);
        # templates come from cfg (or the defaults if no config was given).
        cfg = cfg or _DEFAULT_GAME
        f = self.fragments(window, cfg)
        # Without notes — the plain transcript of past rounds (respecting the window).
        if self.notes is None:
            return [Message("user", f["recent_rounds"])] if f["recent_rounds"] else []
        # With notes: notes_view (the consolidated notes), then — only when rounds were played
        # since the last consolidation — the notes_buffer section with those rounds rendered raw
        # (instead of the full history). The window only limits the buffer.
        # {lines} — the folded rounds' compact lines (dropped with its line when history_line is
        # unset). Replaced first: the note text must not be rescanned for {lines}.
        # {notes_line} — the saved notes rendered THROUGH msg_self (the tag stays in sync with the
        # agent's own lines); {notes} — the raw text, for custom layouts.
        parts = [_fill(cfg.notes_view, "{lines}", f["history_lines"])
                 .replace("{notes_line}", cfg.msg_self.replace("{text}", self.notes))
                 .replace("{notes}", self.notes)]
        if f["recent_rounds"]:
            parts.append(cfg.notes_buffer.replace("{buffer}", f["recent_rounds"]))
        return [Message("user", "\n\n".join(parts))]


def render_turns(transcript: list[dict], me_id: str, msg_self: str, msg_partner: str) -> str:
    """Render cheap-talk turns with <you>/<name> tags — shared code for history and the live feed.

    Args:
        transcript: The round's turns (each with `speaker` and `text`).
        me_id: Identifier of the viewer (their turns are rendered as <you>).
        msg_self: Template for one's own turn ({text}).
        msg_partner: Template for the partner's turn ({partner}/{text}).

    Returns:
        Turns joined by newlines (empty string if there are none).
    """
    lines = []
    for t in transcript:
        speaker = t.get("speaker")
        text = t.get("text", "")
        if speaker == me_id:
            lines.append(msg_self.replace("{text}", text))
        else:
            lines.append(msg_partner.replace("{partner}", speaker).replace("{text}", text))
    return "\n".join(lines)


def _fill(template: str, placeholder: str, value: str) -> str:
    """Substitute a placeholder; an empty value removes the placeholder's whole line."""
    if value:
        return template.replace(placeholder, value)
    return template.replace(placeholder + "\n", "").replace(placeholder, "")


def _window(entries: list[MemoryEntry], window: int | None) -> list[MemoryEntry]:
    if window is None:
        return entries
    if window <= 0:
        return []
    return entries[-window:]


def _render_entry(e: MemoryEntry, cfg: GameCfg, template: str | None = None) -> str:
    # One past round = one full-round template (cfg.history_prompt by default; the compact
    # cfg.history_line for rounds folded into notes) filled by placeholder. {feed} is the rendered
    # cheap-talk of the round (same as the live prompts' {feed}; its whole line is dropped when
    # the round had none); the rest are field substitutions. What the response/takeaway look like
    # is baked into the experiment's own template, so the code does not branch here.
    tmpl = template or cfg.history_prompt
    feed = render_turns(e.transcript, e.my_id, cfg.msg_self, cfg.msg_partner) if e.transcript else ""
    body = _fill(tmpl, "{feed}", feed)
    reason = cfg.reason_agreed if _both_agreed(e) else cfg.reason_limit
    return (
        body
        # {my_number_line} — the agent's own number rendered THROUGH msg_self (so the tag stays in
        # sync with the cheap-talk lines). Must run before {my_number} ({my_number} is its substring).
        .replace("{my_number_line}", cfg.msg_self.replace("{text}", str(e.my_number)))
        .replace("{reason}", reason)
        .replace("{my_rationale}", e.my_rationale or "")
        .replace("{my_number}", str(e.my_number))
        .replace("{partner_number}", str(e.partner_number))
        .replace("{partner_payoff}", f"{e.partner_payoff:g}")
        .replace("{partner}", e.partner_id)
        .replace("{payoff}", f"{e.payoff:g}")
        .replace("{round}", str(e.round))
        .replace("{total}", f"{e.score + e.payoff:g}")
        .replace("{my_reflection}", e.my_reflection or "")
    )


def both_agreed(transcript: list[dict], a_id: str, b_id: str) -> bool:
    """Whether the chat closed by agreement: both participants set finish/ready=true at least once.

    Otherwise the chat hit the turn limit. Single source of truth for the close line — both
    in the history of past rounds (memory) and in the live DECIDE phase (reputation_pd)."""
    ready_speakers = {t.get("speaker") for t in transcript if t.get("ready")}
    return {a_id, b_id} <= ready_speakers


def _both_agreed(e: MemoryEntry) -> bool:
    return both_agreed(e.transcript, e.my_id, e.partner_id)
