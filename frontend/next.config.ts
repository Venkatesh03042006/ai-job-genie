import path from "path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Pins the workspace root to this directory - a stray lockfile in the
  // user's home dir (outside this project) otherwise makes Turbopack guess
  // wrong and warn on every dev server start.
  turbopack: {
    root: path.join(__dirname),
  },
};

export default nextConfig;
