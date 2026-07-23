"use client";

import { useEffect, useState } from "react";
import { apiFetch } from "@/app/lib/api";

interface HistoryMatch {
  rank: number;
  job_doc_id: string;
  category: string;
  title: string;
  score: number;
}

interface HistoryEntry {
  id: number;
  original_filename: string | null;
  submitted_at: string;
  ats_score: number | null;
  skill_gap: string[];
  matches: HistoryMatch[];
}

/** Shown to logged-in users - refetches whenever refreshKey changes, so the
 * parent can trigger a reload right after a new match submission.
 */
export default function MatchHistory({ refreshKey }: { refreshKey: number }) {
  const [history, setHistory] = useState<HistoryEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    apiFetch("/api/match/history/")
      .then(async (res) => {
        if (!res.ok) throw new Error(`status ${res.status}`);
        return res.json();
      })
      .then((data: HistoryEntry[]) => {
        if (!cancelled) setHistory(data);
      })
      .catch(() => {
        if (!cancelled) setError("Couldn't load match history.");
      });

    return () => {
      cancelled = true;
    };
  }, [refreshKey]);

  if (error) return <p className="text-sm text-red-700">{error}</p>;
  if (history === null) return <p className="text-sm text-gray-500">Loading history…</p>;
  if (history.length === 0) {
    return <p className="text-sm text-gray-500">No past submissions yet.</p>;
  }

  return (
    <div className="flex flex-col gap-3">
      {history.map((entry) => (
        <div key={entry.id} className="border border-gray-200 rounded p-3 text-sm">
          <div className="flex items-baseline justify-between gap-2 text-gray-600">
            <span>{entry.original_filename ?? "Pasted text"}</span>
            <span className="text-xs">
              {new Date(entry.submitted_at).toLocaleString()}
            </span>
          </div>
          {entry.ats_score !== null && (
            <p className="mt-1 text-gray-600">
              ATS score: <span className="font-medium">{entry.ats_score.toFixed(1)}%</span>
              {entry.skill_gap.length > 0 && ` · ${entry.skill_gap.length} missing keywords`}
            </p>
          )}
          <ul className="mt-1 flex flex-col gap-0.5">
            {entry.matches.map((m) => (
              <li key={m.job_doc_id}>
                #{m.rank} {m.title} — score {m.score.toFixed(3)}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
