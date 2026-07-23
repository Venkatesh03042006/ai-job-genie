export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

function getCookie(name: string): string | null {
  const match = document.cookie
    .split("; ")
    .find((row) => row.startsWith(`${name}=`));
  return match ? decodeURIComponent(match.split("=").slice(1).join("=")) : null;
}

/** fetch() wrapper that always sends the Django session cookie and, for
 * unsafe methods, the CSRF token DRF's SessionAuthentication requires once a
 * user is logged in (see accounts/views.py's MeView, which seeds the
 * csrftoken cookie via @ensure_csrf_cookie).
 */
export async function apiFetch(path: string, init: RequestInit = {}) {
  const method = (init.method ?? "GET").toUpperCase();
  const headers = new Headers(init.headers);

  if (method !== "GET" && method !== "HEAD") {
    const csrfToken = getCookie("csrftoken");
    if (csrfToken) headers.set("X-CSRFToken", csrfToken);
  }

  return fetch(`${API_URL}${path}`, {
    ...init,
    method,
    headers,
    credentials: "include",
  });
}
