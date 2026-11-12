import json
import subprocess
from uuid import uuid4

import httpx

from .agents import invoke
from .store import defer_busy_claim, now


class Coordinator:
    def __init__(self, vault, agents):
        self.vault = vault
        self.agents = agents
        with vault.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, status TEXT NOT NULL,
                agent TEXT NOT NULL, question TEXT NOT NULL, transcript TEXT NOT NULL,
                answer TEXT, error TEXT, created TEXT NOT NULL, updated TEXT NOT NULL)""")

    def submit(self, agent, question, owner="local"):
        if agent not in self.agents:
            raise ValueError(f"Agent {agent!r} is not enabled; available: {list(self.agents)}")
        if not isinstance(question, str) or not 1 <= len(question) <= 8000:
            raise ValueError("question must contain 1 to 8000 characters")
        job_id = uuid4().hex
        with self.vault.connect() as db:
            db.execute(
                "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?)",
                (job_id, owner, "queued", agent, question, "[]", None, None, now(), now()),
            )
        return self.get(job_id, owner)

    def get(self, job_id, owner="local"):
        with self.vault.connect() as db:
            row = db.execute(
                "SELECT * FROM jobs WHERE id=? AND owner=?", (job_id, owner)
            ).fetchone()
        if row is None:
            raise ValueError("Task does not exist for this client")
        result = dict(row)
        result["transcript"] = json.loads(result["transcript"])
        return result

    def recover(self):
        with self.vault.connect() as db:
            db.execute(
                "UPDATE jobs SET status='interrupted',error=?,updated=? WHERE status='running'",
                ("Server stopped during a live call; not retried to avoid duplicate usage", now()),
            )

    def _claim(self, job_id):
        with defer_busy_claim(), self.vault.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM jobs WHERE status='queued' AND (? IS NULL OR id=?) "
                "ORDER BY created LIMIT 1",
                (job_id, job_id),
            ).fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='running',updated=? WHERE id=?", (now(), row["id"]))
            return dict(row)
        return None

    def run_next(self, max_turns=6, timeout=None, job_id=None, peer_challenge=None):
        job = self._claim(job_id)
        if job is None:
            return False
        transcript = []
        stack = []
        current = job["agent"]
        question = job["question"]
        try:
            for _ in range(max_turns):
                evidence = self.vault.search(question, scopes=("shared",), limit=6)
                envelope = {
                    "contract": (
                        "Answer the current question using the reviewed preferences and quoted "
                        "memory evidence. Cite file paths and headings for memory facts. Say when "
                        "evidence is absent or conflicting. Return JSON with answer and delegate. "
                        "delegate is null to finish this turn, or {agent,question} to ask a peer. "
                        "A peer's completed reply will be returned to its requester automatically. "
                        "Do not ask a peer again once its reply answers the question. Only "
                        "registered peers can be called. No file writes or external actions."
                    ),
                    "agent": current,
                    "available_peers": [name for name in self.agents if name != current],
                    "reviewed_preferences": self.vault.core(),
                    "profile": self.vault.get("profile/PROFILE.md"),
                    "original_question": job["question"],
                    "current_question": question,
                    "quoted_memory_evidence": evidence,
                    "actual_previous_replies": transcript,
                }
                if peer_challenge is not None and current == peer_challenge[0]:
                    # Only the peer sees the challenge before its actual reply is recorded.
                    envelope["verification_challenge"] = peer_challenge[1]
                    envelope["contract"] += (
                        " For this live verification, return verification_challenge verbatim "
                        "in your answer with delegate null."
                    )
                reply = invoke(self.agents[current], envelope, self.vault.state, timeout)
                transcript.append(
                    {
                        "agent": current,
                        "question": question,
                        **reply,
                        "memory_sources": sorted({e["path"] for e in evidence}),
                        "at": now(),
                    }
                )
                with self.vault.connect() as db:
                    db.execute(
                        "UPDATE jobs SET transcript=?,updated=? WHERE id=?",
                        (json.dumps(transcript, ensure_ascii=False), now(), job["id"]),
                    )
                if reply["delegate"]:
                    peer = reply["delegate"]["agent"]
                    if peer not in self.agents or peer == current:
                        raise ValueError(f"Invalid handoff from {current!r} to {peer!r}")
                    stack.append((current, question))
                    current, question = peer, reply["delegate"]["question"]
                    continue
                if stack:
                    current, question = stack.pop()
                    question += (
                        "\nYour peer has replied. Use the actual transcript and finish if possible."
                    )
                    continue
                with self.vault.connect() as db:
                    db.execute(
                        "UPDATE jobs SET status='done',answer=?,updated=? WHERE id=?",
                        (reply["answer"], now(), job["id"]),
                    )
                return True
            raise ValueError(f"Handoff limit reached after {max_turns} agent turns")
        except (
            ValueError,
            KeyError,
            OSError,
            RuntimeError,
            subprocess.TimeoutExpired,
            httpx.HTTPError,
        ) as err:
            with self.vault.connect() as db:
                db.execute(
                    "UPDATE jobs SET status='failed',error=?,updated=? WHERE id=?",
                    (f"{type(err).__name__}: {err}", now(), job["id"]),
                )
            return True
