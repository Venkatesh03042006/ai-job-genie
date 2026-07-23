"use client";

import { useEffect, useState } from "react";
import { apiFetch, API_URL } from "@/app/lib/api";
import AuthPanel, { AuthUser } from "@/app/components/AuthPanel";
import MatchHistory from "@/app/components/MatchHistory";

interface Explanation {
  resume_chunk: string;
  jd_chunk: string;
  score: number;
}

interface Match {
  rank: number;
  job_doc_id: string;
  category: string;
  title: string;
  score: number;
  explanations: Explanation[];
}

interface MatchResponse {
  resume_id: number;
  matches: Match[];
  ats_score: number;
  skill_gap: string[];
}

const MIN_RESUME_LENGTH = 10;
const MAX_UPLOAD_SIZE = 5 * 1024 * 1024; // 5MB
const ALLOWED_UPLOAD_EXTENSIONS = [".pdf", ".docx", ".txt"];
const SKILL_GAP_PREVIEW_COUNT = 30;

type Mode = "paste" | "upload";

function parseErrorBody(body: unknown): string | null {
  if (body && typeof body === "object") {
    const record = body as Record<string, unknown>;
    const fieldError = record.resume_text ?? record.resume_file;
    if (fieldError) {
      return Array.isArray(fieldError) ? fieldError.join(" ") : String(fieldError);
    }
    if (record.detail) {
      return String(record.detail);
    }
  }
  return null;
}

export default function Home() {
  const [mode, setMode] = useState<Mode>("paste");
  const [resumeText, setResumeText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [matches, setMatches] = useState<Match[] | null>(null);
  const [atsScore, setAtsScore] = useState<number | null>(null);
  const [skillGap, setSkillGap] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [user, setUser] = useState<AuthUser | null>(null);
  const [historyRefreshKey, setHistoryRefreshKey] = useState(0);

  useEffect(() => {
    apiFetch("/api/auth/me/")
      .then((res) => res.json())
      .then((data: { user: AuthUser | null }) => setUser(data.user))
      .catch(() => setUser(null));
  }, []);

  async function submitMatch(path: string, body: BodyInit, headers?: HeadersInit) {
    setLoading(true);
    setError(null);

    try {
      const res = await apiFetch(path, { method: "POST", headers, body });

      if (!res.ok) {
        let message = `Request failed (status ${res.status}).`;
        try {
          const parsed = parseErrorBody(await res.json());
          if (parsed) message = parsed;
        } catch {
          // response wasn't JSON - keep the generic status message
        }
        setError(message);
        setMatches(null);
        setAtsScore(null);
        setSkillGap([]);
        return;
      }

      const data: MatchResponse = await res.json();
      setMatches(data.matches);
      setAtsScore(data.ats_score);
      setSkillGap(data.skill_gap);
      setHistoryRefreshKey((k) => k + 1);
    } catch {
      setError(
        `Couldn't reach the server - is the Django backend running at ${API_URL}?`
      );
      setMatches(null);
      setAtsScore(null);
      setSkillGap([]);
    } finally {
      setLoading(false);
    }
  }

  async function handlePasteSubmit(e: React.FormEvent) {
    e.preventDefault();

    if (resumeText.trim().length < MIN_RESUME_LENGTH) {
      setError(`Resume text must be at least ${MIN_RESUME_LENGTH} characters.`);
      return;
    }

    await submitMatch(
      "/api/match/",
      JSON.stringify({ resume_text: resumeText }),
      { "Content-Type": "application/json" }
    );
  }

  async function handleUploadSubmit(e: React.FormEvent) {
    e.preventDefault();

    if (!file) {
      setError("Please choose a file to upload.");
      return;
    }
    const extension = "." + (file.name.split(".").pop()?.toLowerCase() ?? "");
    if (!ALLOWED_UPLOAD_EXTENSIONS.includes(extension)) {
      setError(
        `Unsupported file type. Please upload a ${ALLOWED_UPLOAD_EXTENSIONS.join(", ")} file.`
      );
      return;
    }
    if (file.size > MAX_UPLOAD_SIZE) {
      setError(
        `File too large (${(file.size / 1024 / 1024).toFixed(1)}MB) - max ${MAX_UPLOAD_SIZE / 1024 / 1024}MB.`
      );
      return;
    }

    const formData = new FormData();
    formData.append("resume_file", file);
    // No Content-Type header - the browser sets the multipart boundary itself.
    await submitMatch("/api/match/upload/", formData);
  }

  return (
    <main className="max-w-3xl mx-auto p-6 flex flex-col gap-6">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <h1 className="text-2xl font-bold">AI Job Genie — Resume Matcher</h1>
        <AuthPanel user={user} onUserChange={setUser} />
      </div>

      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => setMode("paste")}
          className={`px-3 py-1.5 rounded text-sm font-medium ${
            mode === "paste"
              ? "bg-blue-600 text-white"
              : "bg-gray-200 text-gray-800"
          }`}
        >
          Paste text
        </button>
        <button
          type="button"
          onClick={() => setMode("upload")}
          className={`px-3 py-1.5 rounded text-sm font-medium ${
            mode === "upload"
              ? "bg-blue-600 text-white"
              : "bg-gray-200 text-gray-800"
          }`}
        >
          Upload file
        </button>
      </div>

      {mode === "paste" ? (
        <form onSubmit={handlePasteSubmit} className="flex flex-col gap-3">
          <label htmlFor="resume-text" className="font-medium">
            Paste your resume text
          </label>
          <textarea
            id="resume-text"
            value={resumeText}
            onChange={(e) => setResumeText(e.target.value)}
            rows={10}
            placeholder="Paste your career objective, skills, and past positions here..."
            className="border border-gray-400 rounded p-3 font-mono text-sm"
          />
          <button
            type="submit"
            disabled={loading}
            className="self-start bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "Matching…" : "Find matches"}
          </button>
        </form>
      ) : (
        <form onSubmit={handleUploadSubmit} className="flex flex-col gap-3">
          <label htmlFor="resume-file" className="font-medium">
            Upload your resume (.pdf, .docx, .txt)
          </label>
          <input
            id="resume-file"
            type="file"
            accept=".pdf,.docx,.txt"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="border border-gray-400 rounded p-3 text-sm"
          />
          <button
            type="submit"
            disabled={loading}
            className="self-start bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "Uploading & matching…" : "Find matches"}
          </button>
        </form>
      )}

      {error && (
        <div className="border border-red-400 bg-red-50 text-red-800 rounded p-3">
          {error}
        </div>
      )}

      {matches === null && !error && (
        <p className="text-gray-500">No matches yet — submit a resume above.</p>
      )}

      {matches !== null && (
        <div className="flex flex-col gap-4">
          <h2 className="text-lg font-semibold">
            {matches.length} match{matches.length === 1 ? "" : "es"}
          </h2>

          {atsScore !== null && (
            <div className="border border-gray-300 rounded p-4">
              <div className="flex items-baseline justify-between gap-2">
                <h3 className="font-semibold">ATS score (vs. top match)</h3>
                <span className="text-lg font-bold">{atsScore.toFixed(1)}%</span>
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Percentage of the top-matched job&apos;s keywords found in your resume.
              </p>

              {skillGap.length > 0 && (
                <div className="mt-3">
                  <p className="text-sm text-gray-600 mb-1">
                    Missing keywords ({skillGap.length}):
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {skillGap.slice(0, SKILL_GAP_PREVIEW_COUNT).map((kw) => (
                      <span
                        key={kw}
                        className="text-xs bg-gray-200 text-gray-800 rounded px-2 py-0.5"
                      >
                        {kw}
                      </span>
                    ))}
                    {skillGap.length > SKILL_GAP_PREVIEW_COUNT && (
                      <span className="text-xs text-gray-500 px-2 py-0.5">
                        +{skillGap.length - SKILL_GAP_PREVIEW_COUNT} more
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {matches.map((m) => (
            <div key={m.job_doc_id} className="border border-gray-300 rounded p-4">
              <div className="flex items-baseline justify-between gap-2">
                <h3 className="font-semibold">
                  #{m.rank} {m.title}
                </h3>
                <span className="text-sm text-gray-600 whitespace-nowrap">
                  score {m.score.toFixed(3)}
                </span>
              </div>
              <span className="inline-block mt-1 mb-3 text-xs bg-gray-200 text-gray-800 rounded px-2 py-0.5">
                {m.category}
              </span>

              {m.explanations.length === 0 ? (
                <p className="text-sm text-gray-500">
                  No genuine (non-duplicate) chunk match found for this job.
                </p>
              ) : (
                <ul className="flex flex-col gap-2">
                  {m.explanations.map((exp, i) => (
                    <li key={i} className="text-sm border-l-2 border-blue-400 pl-3">
                      <div>
                        <span className="text-gray-500">Your: </span>
                        {exp.resume_chunk}
                      </div>
                      <div>
                        <span className="text-gray-500">Matched: </span>
                        {exp.jd_chunk}
                      </div>
                      <div className="text-xs text-gray-400">
                        chunk score {exp.score.toFixed(3)}
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ))}
        </div>
      )}

      {user && (
        <div className="flex flex-col gap-3 border-t border-gray-200 pt-6">
          <h2 className="text-lg font-semibold">Your match history</h2>
          <MatchHistory refreshKey={historyRefreshKey} />
        </div>
      )}
    </main>
  );
}
