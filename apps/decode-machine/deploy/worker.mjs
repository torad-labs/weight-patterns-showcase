const MOUNT = "/decode-machine";
const HASHED_ASSET = /^\/assets\/.+-[A-Za-z0-9_-]{8,}\.(?:js|css)$/u;
const ALLOWED_PREFIXES = ["/assets/", "/replay/", "/evidence/"];

const SECURITY_HEADERS = {
  "Content-Security-Policy":
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self' blob:; worker-src 'self' blob:; frame-ancestors 'self'; base-uri 'self'; form-action 'self'; object-src 'none'; upgrade-insecure-requests",
  "Permissions-Policy": "geolocation=(), microphone=(), camera=(), browsing-topics=()",
  "Referrer-Policy": "strict-origin-when-cross-origin",
  "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "SAMEORIGIN",
};

function withHeaders(response, cacheControl) {
  const headers = new Headers(response.headers);
  headers.set("Cache-Control", cacheControl);
  for (const [key, value] of Object.entries(SECURITY_HEADERS)) {
    headers.set(key, value);
  }
  return new Response(response.body, {
    headers,
    status: response.status,
    statusText: response.statusText,
  });
}

function notFound() {
  return withHeaders(
    new Response("Not Found", {
      headers: { "Content-Type": "text/plain; charset=utf-8" },
      status: 404,
    }),
    "no-store",
  );
}

function redirect(location) {
  return withHeaders(
    new Response(null, {
      headers: { Location: location },
      status: 301,
    }),
    "public, max-age=300",
  );
}

export default {
  async fetch(request, env) {
    if (request.method !== "GET" && request.method !== "HEAD") {
      return withHeaders(
        new Response("Method Not Allowed", {
          headers: { Allow: "GET, HEAD", "Content-Type": "text/plain; charset=utf-8" },
          status: 405,
        }),
        "no-store",
      );
    }

    const url = new URL(request.url);
    const { pathname, search } = url;

    if (pathname !== MOUNT && !pathname.startsWith(`${MOUNT}/`)) {
      return notFound();
    }
    if (pathname.includes("//") || pathname.includes("..")) {
      return notFound();
    }

    const subpath = pathname.slice(MOUNT.length) || "/";

    if (subpath === "/index.html") {
      return redirect(`${MOUNT}${search}`);
    }
    if (subpath === "/teaching.html" || subpath === "/teaching" || subpath === "/teaching/") {
      return redirect(`${MOUNT}/learning${search}`);
    }

    if (subpath === "/") {
      const assetReq = new Request(new URL("/index.html", url.origin), request);
      const res = await env.ASSETS.fetch(assetReq);
      return res.ok ? withHeaders(res, "public, max-age=60, must-revalidate") : notFound();
    }

    if (subpath === "/learning" || subpath === "/learning/") {
      const assetReq = new Request(new URL("/teaching.html", url.origin), request);
      const res = await env.ASSETS.fetch(assetReq);
      return res.ok ? withHeaders(res, "public, max-age=60, must-revalidate") : notFound();
    }

    const allowedAsset =
      subpath === "/favicon.svg" || ALLOWED_PREFIXES.some((prefix) => subpath.startsWith(prefix));
    if (!allowedAsset) {
      return notFound();
    }

    const assetReq = new Request(new URL(subpath, url.origin), request);
    const res = await env.ASSETS.fetch(assetReq);
    if (!res.ok) {
      return notFound();
    }

    const cacheControl = HASHED_ASSET.test(subpath)
      ? "public, max-age=31536000, immutable"
      : "public, max-age=3600";
    return withHeaders(res, cacheControl);
  },
};
