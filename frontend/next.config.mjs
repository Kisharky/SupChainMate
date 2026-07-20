/** @type {import('next').NextConfig} */
const API_BASE = process.env.API_PROXY_TARGET || "http://localhost:8000";

const nextConfig = {
  reactStrictMode: true,
  // Proxy /api/* to the FastAPI backend in dev so the browser hits one origin
  // and CORS is a non-issue. In production, point API_PROXY_TARGET at the API.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_BASE}/api/:path*` }];
  },
};

export default nextConfig;
