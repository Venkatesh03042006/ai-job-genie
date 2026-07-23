"use client";

import { useState } from "react";
import { apiFetch } from "@/app/lib/api";

export interface AuthUser {
  id: number;
  username: string;
}

interface AuthPanelProps {
  user: AuthUser | null;
  onUserChange: (user: AuthUser | null) => void;
}

type AuthMode = "login" | "register";

function parseAuthError(body: unknown): string {
  if (body && typeof body === "object") {
    const record = body as Record<string, unknown>;
    for (const value of Object.values(record)) {
      if (Array.isArray(value)) return value.join(" ");
      if (typeof value === "string") return value;
    }
  }
  return "Something went wrong. Please try again.";
}

export default function AuthPanel({ user, onUserChange }: AuthPanelProps) {
  const [mode, setMode] = useState<AuthMode>("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const res = await apiFetch(`/api/auth/${mode}/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      if (!res.ok) {
        setError(parseAuthError(await res.json().catch(() => null)));
        return;
      }

      const data: AuthUser = await res.json();
      onUserChange(data);
      setUsername("");
      setPassword("");
    } catch {
      setError("Couldn't reach the server.");
    } finally {
      setLoading(false);
    }
  }

  async function handleLogout() {
    setLoading(true);
    try {
      await apiFetch("/api/auth/logout/", { method: "POST" });
      onUserChange(null);
    } finally {
      setLoading(false);
    }
  }

  if (user) {
    return (
      <div className="flex items-center gap-3 text-sm">
        <span>
          Signed in as <span className="font-medium">{user.username}</span>
        </span>
        <button
          type="button"
          onClick={handleLogout}
          disabled={loading}
          className="px-3 py-1 rounded bg-gray-200 text-gray-800 disabled:opacity-50"
        >
          Log out
        </button>
      </div>
    );
  }

  return (
    <div className="border border-gray-300 rounded p-4 flex flex-col gap-3 max-w-sm">
      <div className="flex gap-2 text-sm">
        <button
          type="button"
          onClick={() => setMode("login")}
          className={`px-2 py-1 rounded font-medium ${
            mode === "login" ? "bg-blue-600 text-white" : "bg-gray-200 text-gray-800"
          }`}
        >
          Log in
        </button>
        <button
          type="button"
          onClick={() => setMode("register")}
          className={`px-2 py-1 rounded font-medium ${
            mode === "register" ? "bg-blue-600 text-white" : "bg-gray-200 text-gray-800"
          }`}
        >
          Register
        </button>
      </div>

      <form onSubmit={handleSubmit} className="flex flex-col gap-2">
        <input
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
          className="border border-gray-400 rounded p-2 text-sm"
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          className="border border-gray-400 rounded p-2 text-sm"
        />
        <button
          type="submit"
          disabled={loading}
          className="bg-blue-600 text-white px-3 py-2 rounded text-sm disabled:opacity-50"
        >
          {loading ? "Please wait…" : mode === "login" ? "Log in" : "Create account"}
        </button>
      </form>

      {error && <p className="text-sm text-red-700">{error}</p>}
    </div>
  );
}
